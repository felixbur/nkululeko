"""Shared parts of the recurrent models (lstm, lstm_reg).

The models take frame sequences from the feature pipeline
(FEATS.level = frame, index (file, start, end, frame), see
nkululeko.utils.sequences) and one label per segment. Sequences keep
their own length: a batch is zero-padded to its longest sequence, the
padding is skipped by the RNN (packed sequences) and left out of the mean
over time.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from nkululeko.models.model import Model
from nkululeko.models.rnn_config import RNNConfig
from nkululeko.optimizers import (
    get_optimizer,
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)
from nkululeko.utils.random_seed import parse_seed
from nkululeko.utils.sequences import is_sequence, to_arrays


class SequenceDataset(Dataset):
    """One (frames, label) item per row of df, the frames taken from the
    feature sequences of the same segment."""

    def __init__(self, df, feats, target, label_dtype, domain_labels=None):
        index, arrays = to_arrays(feats)
        by_segment = {_key(seg): arr for seg, arr in zip(index, arrays)}
        missing = [seg for seg in df.index if _key(seg) not in by_segment]
        if missing:
            raise ValueError(
                f"{len(missing)} sample(s) have no feature frames, e.g. {missing[0]}"
            )
        self.frames = [torch.from_numpy(by_segment[_key(seg)]) for seg in df.index]
        self.labels = df[target].to_numpy().astype(label_dtype)
        # (len(df), n_dann_columns) int array; only the train loader sets
        # it, then items have a third element
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if self.domain_labels is None:
            return self.frames[idx], self.labels[idx]
        return (
            self.frames[idx],
            self.labels[idx],
            torch.from_numpy(self.domain_labels[idx]),
        )


def _key(segment):
    """Segment key usable in a dict: NaT becomes None."""
    return tuple(None if pd.isna(x) else x for x in segment)


def collate(batch):
    """Zero-pad the sequences of a batch to the longest one.

    Returns (frames (batch, T, F), lengths (batch,), labels[, domain
    labels])."""
    frames = [item[0] for item in batch]
    lengths = torch.tensor([len(f) for f in frames])
    padded = nn.utils.rnn.pad_sequence(frames, batch_first=True)
    labels = torch.as_tensor(np.array([item[1] for item in batch]))
    if len(batch[0]) == 2:
        return padded, lengths, labels
    return padded, lengths, labels, torch.stack([item[2] for item in batch])


class RNNNet(nn.Module):
    """A stacked recurrent network over a frame sequence, mean-pooled over
    the real (unpadded) frames, then a linear output layer.

    cell_cls is a parameter so that a later GRU type can reuse the class.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        out_dim,
        bidirectional,
        dropout,
        cell_cls=nn.LSTM,
    ):
        super().__init__()
        self.rnn = cell_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # the cells ignore (and warn about) dropout with a single layer
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.feat_dim = hidden_size * (2 if bidirectional else 1)
        self.out = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x, lengths, return_features=False):
        # x: (batch, T, input_size), zero-padded after each sequence's end
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=x.shape[1]
        )
        mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            < lengths.to(x.device)[:, None]
        ).unsqueeze(-1)
        pooled = (outputs * mask).sum(dim=1) / lengths.to(x.device)[:, None]
        out = self.out(pooled)
        if return_features:
            return out, pooled
        return out


class RNNModel(Model):
    """Base class of the recurrent models: network, data loading and the
    training epoch. Subclasses set out_dim, label_dtype and the criterion
    and implement the evaluation."""

    input_kind = "sequence"
    type_name = None  # MODEL.type, set by the subclass
    label_dtype = np.int64
    cell_cls = nn.LSTM

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("ann")
        self.name = self.type_name
        self.target = self.context.config["DATA"]["target"]
        if not is_sequence(feats_train):
            self.util.error(
                f"MODEL.type = {self.name} needs frame sequences: set "
                "FEATS.level = frame and use a feature type with frame "
                "output (e.g. FEATS.type = ['os'])"
            )

        seed_value = self.util.config_val("MODEL", "random_seed", "False")
        try:
            manual_seed = parse_seed(seed_value)
        except ValueError:
            self.util.error(
                f"MODEL.random_seed must be an integer or False (got '{seed_value}')"
            )
        if manual_seed is not None:
            self.util.debug(f"seeding random to {manual_seed}")
            torch.manual_seed(manual_seed)

        self._setup_task()
        self.cfg = RNNConfig.from_util(self.util)
        self.device = self.cfg.device
        self.util.debug(
            f"{self.name}: hidden_size={self.cfg.hidden_size}, "
            f"num_layers={self.cfg.num_layers}, "
            f"bidirectional={self.cfg.bidirectional}"
        )

        self.net = RNNNet(
            feats_train.shape[1],
            self.cfg.hidden_size,
            self.cfg.num_layers,
            self.out_dim,
            self.cfg.bidirectional,
            self.cfg.dropout,
            cell_cls=self.cell_cls,
        ).to(self.device)
        self._init_dann(df_train, self.net.feat_dim, self.cfg.dann)
        self.trainloader = self.get_loader(df_train, feats_train, True, train=True)
        self.testloader = self.get_loader(df_test, feats_test, False)

        params = self._dann_params(self.net)
        self.optimizer, self.learning_rate = get_optimizer(
            params, self.util, default_lr=1e-3, default_optimizer="adamw"
        )
        self.scheduler, self.scheduler_type, self.scheduler_needs_init = get_scheduler(
            self.optimizer, self.util, default_scheduler="cosine"
        )

    def _setup_task(self):
        """Set self.criterion and self.out_dim."""
        raise NotImplementedError

    def get_loader(self, df, feats, shuffle, train=False):
        # MODEL.nan_strategy, as for the other models; fills frames, never
        # drops them
        feats = self._handle_model_nan(feats, f"{self.name}, frames")
        dataset = SequenceDataset(
            df,
            feats,
            self.target,
            self.label_dtype,
            domain_labels=self._domain_labels(df) if train else None,
        )
        if shuffle:  # the train loader
            return self._make_train_loader(
                dataset, df, self.cfg.batch_size, collate_fn=collate
            )
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collate
        )

    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df
        self.testloader = self.get_loader(data_df, feats_df, False)

    def reset_test(self, df_test, feats_test):
        self.set_testdata(df_test, feats_test)

    def train(self):
        """Train for one epoch (modelrunner runs the epoch loop)."""
        if self.scheduler_needs_init and self.scheduler is None:
            self.scheduler = initialize_cosine_scheduler(
                self.optimizer, self.util, steps_per_epoch=len(self.trainloader)
            )
            self.scheduler_needs_init = False

        self.net.train()
        losses = []
        for batch in self.trainloader:
            frames, lengths, labels = batch[0], batch[1], batch[2]
            domain_labels = None
            if self.dann_heads is not None:
                domain_labels = batch[3].to(self.device)
            frames, labels = frames.to(self.device), labels.to(self.device)

            loss = self._optimize(
                self.optimizer,
                lambda: self._forward_loss(frames, lengths, labels, domain_labels),
                skip_non_finite=True,  # ccc/pcc are NaN on a one-value batch
                modules=[self.net],
            )
            if loss is not None:  # None: a skipped (non-finite) step
                losses.append(loss.item())
                # skipped steps don't advance the learning-rate schedule
                step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=True)

        step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=False)
        self.loss = float(np.mean(losses)) if losses else 0.0

    def _task_loss(self, out, labels):
        return self.criterion(out, labels)

    def _forward_loss(self, frames, lengths, labels, domain_labels):
        """Task loss, plus the domain losses of the DANN heads."""
        if self.dann_heads is None:
            return self._task_loss(self.net(frames, lengths), labels)
        out, hidden = self.net(frames, lengths, return_features=True)
        return self._task_loss(out, labels) + self.dann_heads.loss(
            hidden, domain_labels
        )

    def _run(self, loader):
        """Network outputs and labels for every sample of loader, in
        order, and the evaluation loss: the batch losses weighted by batch
        size, leaving out non-finite ones."""
        self.net.eval()
        outputs, targets, losses, counts = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                frames, lengths, labels = batch[0], batch[1], batch[2]
                out = self.net(frames.to(self.device), lengths).cpu()
                loss = self._task_loss(out, labels)
                if torch.isfinite(loss):
                    losses.append(loss.item())
                    counts.append(len(labels))
                outputs.append(out)
                targets.append(labels)
        loss_eval = float(np.average(losses, weights=counts)) if losses else 0.0
        if not outputs:
            return torch.zeros(0, self.out_dim), torch.zeros(0), loss_eval
        return torch.cat(outputs), torch.cat(targets), loss_eval

    def store(self):
        torch.save(self.net.state_dict(), self.store_path)

    def load(self, run, epoch):
        self.set_id(run, epoch)
        try:
            self.net.load_state_dict(
                torch.load(self.store_path, map_location=self.device, weights_only=True)
            )
        except FileNotFoundError:
            self.util.error(f"model file not found: {self.store_path}")
        self.net.eval()
