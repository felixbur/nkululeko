"""LSTMModel: [MODEL] type = lstm.

Frame-sequence LSTM for classification (GH #256). Like model_aasist.py, it
reads raw audio directly from df_train/df_test (via audiofile.read()) and
bypasses nkululeko's FeatureExtractor entirely, so a config using this
model must set FEATS.type = []. Unlike model_aasist.py it does not feed an
SSL frontend: it computes a frame sequence (log-mel spectrogram, or
opensmile LLDs -- see RNN.feature_source in rnn_config.py) from a
fixed-length (padded-by-tiling) waveform, and trains an LSTM over that
sequence from scratch. model_type is set to "ann" (not "finetuned"):
train() runs exactly one epoch per call, so modelrunner.do_epochs()'s own
per-epoch loop (patience, dev/test split, ...) applies identically to how
it does for model_adm.py/model_aasist.py.

Unlike AasistModel, this model is not restricted to binary classification
-- self.class_num comes from the experiment's actual label set, same as
MLPModel, since EmoDB-style categorical targets are the point.
"""

import warnings

import audresample
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset

from nkululeko.models.model import Model
from nkululeko.models.rnn_config import RNNConfig
from nkululeko.optimizers import (
    get_optimizer,
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.random_seed import parse_seed

# Frame sources are computed at a fixed sample rate so win/hop (in samples)
# and max_len mean the same duration regardless of the source file's own
# native rate.
_TARGET_SR = 16000


def _pad_or_tile(signal, max_len):
    """Pad-by-tiling (upstream's convention) or truncate to a fixed length.

    An empty signal (e.g. a zero-duration segment) cannot be tiled; it is
    replaced by silence of max_len samples. Duplicated from
    model_aasist.py's identical helper rather than imported -- small
    enough that duplication is cheaper than a new cross-model dependency.
    """
    sig_len = signal.shape[0]
    if sig_len == 0:
        return np.zeros(max_len, dtype=signal.dtype)
    if sig_len >= max_len:
        return signal[:max_len]
    num_repeats = int(max_len / sig_len) + 1
    return np.tile(signal, num_repeats)[:max_len]


class _FrameSequenceDataset(Dataset):
    """Reads raw waveforms directly from disk and turns each one into a
    frame-level sequence, bypassing FeatureExtractor entirely.

    Every waveform is padded-by-tiling/truncated to the same max_len
    before framing (see _pad_or_tile), so every item's frame count T is
    identical regardless of the original segment's duration -- the
    default DataLoader collate can stack them without a custom
    collate_fn/pack_padded_sequence.
    """

    def __init__(self, df, target, cfg, domain_labels=None):
        self.df = df
        self.target = target
        self.cfg = cfg
        # (len(df), n_dann_columns) int array -- when set, items are
        # (frames, label, domain_labels) instead of (frames, label). Only
        # the train loader sets it.
        self.domain_labels = domain_labels

        if cfg.feature_source == "melspec":
            import torchaudio

            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=_TARGET_SR,
                n_fft=cfg.win_samples,
                win_length=cfg.win_samples,
                hop_length=cfg.hop_samples,
                n_mels=cfg.n_mels,
            )
            self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
            self.smile = None
        else:  # "opensmile_lld"
            import opensmile

            self.mel = None
            self.smile = opensmile.Smile(
                feature_set=getattr(opensmile.FeatureSet, cfg.opensmile_set),
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )

    def __len__(self):
        return len(self.df)

    def _frames(self, waveform):
        """waveform: 1-D float32 array of exactly cfg.max_len samples.
        Returns a (T, F) float32 tensor -- F is n_mels for melspec, or the
        opensmile feature set's own LLD dimensionality for opensmile_lld.
        """
        if self.mel is not None:
            signal = torch.from_numpy(waveform)
            spec = self.db(self.mel(signal))  # (n_mels, T), freq-major
            return spec.transpose(0, 1).float()  # (T, n_mels)
        # opensmile already returns a time-major (T, F) frame table.
        result = self.smile.process_signal(waveform, _TARGET_SR)
        return torch.from_numpy(result.to_numpy()).float()

    def __getitem__(self, idx):
        import audiofile

        file, start, end = self.df.index[idx]
        if pd.isna(end):
            signal, sr = audiofile.read(file, offset=start, always_2d=True)
        else:
            signal, sr = audiofile.read(
                file, duration=end - start, offset=start, always_2d=True
            )
        signal = np.asarray(signal).mean(axis=0)  # (channels, samples) -> mono
        if sr != _TARGET_SR:
            signal = audresample.resample(signal, sr, _TARGET_SR).squeeze()

        if signal.shape[0] == 0:
            warnings.warn(
                f"lstm: no audio read from {file} (start={start}, end={end}); "
                "using silence for this sample",
                stacklevel=2,
            )
        signal = _pad_or_tile(signal, self.cfg.max_len).astype(np.float32)
        frames = self._frames(signal)

        row = self.df.iloc[idx]
        label = row[self.target]
        if self.domain_labels is None:
            return frames, label
        return frames, label, torch.from_numpy(self.domain_labels[idx])


class _RNNNet(nn.Module):
    """A stacked RNN cell (LSTM for this model type) over a frame
    sequence, mean-pooled over time, then a linear classifier head.

    cell_cls is a constructor argument (not hardcoded to nn.LSTM) so a
    future sibling model type (e.g. GRU) can reuse this exact class --
    see rnn_config.py's module docstring.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
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
            # nn.LSTM/nn.GRU warn (and ignore it) if dropout > 0 with
            # num_layers == 1 -- there is no inter-layer connection to
            # apply it to.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.feat_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_features=False):
        # x: (batch, T, input_size)
        outputs, _ = self.rnn(x)  # (batch, T, feat_dim)
        # Mean-pooled over time, not the last timestep: _pad_or_tile tiles
        # (repeats) rather than zero-pads, so there is no real/padding
        # boundary to respect, and the last timestep is an arbitrary
        # repeat-point artifact rather than necessarily the most
        # informative frame. Mean-pooling also needs no special-casing
        # for the bidirectional case (no h_n concatenation).
        pooled = outputs.mean(dim=1)  # (batch, feat_dim)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits


class LSTMModel(Model):
    """LSTM = frame-sequence recurrent network for classification."""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("ann")
        self.name = "lstm"
        self.target = self.context.config["DATA"]["target"]

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

        labels = self.context.labels
        self.class_num = len(labels)
        self._setup_criterion()

        self.cfg = RNNConfig.from_util(self.util)
        self.device = self.cfg.device
        self.util.debug(
            f"lstm: feature_source={self.cfg.feature_source}, "
            f"hidden_size={self.cfg.hidden_size}, num_layers={self.cfg.num_layers}"
        )

        self.trainloader = self.get_loader(df_train, shuffle=True, train=True)
        self.testloader = self.get_loader(df_test, shuffle=False)

        # input_size is read empirically off one real sample rather than
        # hardcoded, mirroring model_mlp.py's feats_train.shape[1] -- F
        # depends on feature_source (n_mels for melspec, opensmile's own
        # LLD dimensionality otherwise).
        input_size = self.trainloader.dataset[0][0].shape[-1]
        self.net = _RNNNet(
            input_size,
            self.cfg.hidden_size,
            self.cfg.num_layers,
            self.class_num,
            self.cfg.bidirectional,
            self.cfg.dropout,
        ).to(self.device)

        self._init_dann(df_train, self.net.feat_dim, self.cfg.dann)
        if self.dann_heads is not None:
            # The train loader above was built before the heads existed
            # (it was needed to read input_size), so rebuild it to yield
            # the domain labels.
            self.trainloader = self.get_loader(df_train, shuffle=True, train=True)
        params = self._dann_params(self.net)
        # Higher default LR than AASIST's SSL-fine-tuning 1e-5: this net
        # trains from scratch, closer to ADM's/MLP's training story.
        self.optimizer, self.learning_rate = get_optimizer(
            params, self.util, default_lr=1e-3, default_optimizer="adamw"
        )
        self.scheduler, self.scheduler_type, self.scheduler_needs_init = get_scheduler(
            self.optimizer, self.util, default_scheduler="cosine"
        )

    def get_loader(self, df, shuffle, train=False):
        dataset = _FrameSequenceDataset(
            df,
            self.target,
            self.cfg,
            domain_labels=self._domain_labels(df) if train else None,
        )
        # Each __getitem__ does its own audio read + frame computation --
        # CPU-bound work that a single-process loader (num_workers=0)
        # serializes with GPU compute. MODEL.n_jobs (already read by the
        # base Model class into self.n_jobs) parallelizes it.
        loader_kwargs = {}
        if self.n_jobs > 0:
            loader_kwargs["num_workers"] = self.n_jobs
            loader_kwargs["persistent_workers"] = True
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, **loader_kwargs
        )

    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df
        self.testloader = self.get_loader(data_df, shuffle=False)

    def reset_test(self, df_test, feats_test):
        self.df_test, self.feats_test = df_test, feats_test
        self.testloader = self.get_loader(df_test, shuffle=False)

    def train(self):
        """Train for exactly one epoch (do_epochs() calls this once per
        epoch and owns the epoch loop/patience, matching model_adm.py/
        model_aasist.py)."""
        if self.scheduler_needs_init and self.scheduler is None:
            self.scheduler = initialize_cosine_scheduler(
                self.optimizer, self.util, steps_per_epoch=len(self.trainloader)
            )
            self.scheduler_needs_init = False

        self.net.train()
        losses = []
        for batch in self.trainloader:
            if self.dann_heads is not None:
                frames, labels, domain_labels = batch
                domain_labels = domain_labels.to(self.device)
            else:
                frames, labels = batch
                domain_labels = None
            frames = frames.to(self.device)
            labels = labels.long().to(self.device)

            loss = self._optimize(
                self.optimizer,
                lambda: self._forward_loss(frames, labels, domain_labels),
                modules=[self.net],
            )

            if loss is not None:  # None: SAM skipped a non-finite step
                losses.append(loss.item())
                # a skipped step is not counted, so that the learning-rate
                # schedule follows the weight updates that were applied
                step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=True)

        step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=False)
        self.loss = float(np.mean(losses)) if losses else 0.0

    def _forward_loss(self, frames, labels, domain_labels):
        """Task loss, plus one weighted domain-classification loss per
        DANN head when MODEL.dann_columns is set."""
        if self.dann_heads is None:
            return self.criterion(self.net(frames), labels)
        logits, hidden = self.net(frames, return_features=True)
        return self.criterion(logits, labels) + self.dann_heads.loss(
            hidden, domain_labels
        )

    def evaluate(self, loader):
        logits = torch.zeros(len(loader.dataset), self.class_num)
        targets = torch.zeros(len(loader.dataset))
        self.net.eval()
        losses = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                # the DANN train loader has a third item (domain labels)
                frames, labels = batch[0], batch[1]
                start_index = index * loader.batch_size
                end_index = min((index + 1) * loader.batch_size, len(loader.dataset))
                logits[start_index:end_index, :] = self.net(frames.to(self.device))
                targets[start_index:end_index] = labels
                loss = self.criterion(
                    logits[start_index:end_index, :].to(self.device),
                    labels.to(self.device, dtype=torch.int64),
                )
                losses.append(loss.item())

        loss_eval = float(np.mean(losses)) if losses else 0.0
        predictions = logits.argmax(dim=1)
        uar = (
            recall_score(targets.numpy(), predictions.numpy(), average="macro")
            if len(targets)
            else 0.0
        )
        return uar, targets, predictions, logits, loss_eval

    def get_probas(self, logits):
        # softmax first, since logits aren't themselves probabilities.
        probs = torch.softmax(logits, dim=1).numpy()
        # Columns follow the network's outputs, i.e. every encoded class
        # 0..class_num-1, even if the test split lacks some of them.
        proba_d = {c: list(probs.T[c]) for c in range(self.class_num)}
        probas = pd.DataFrame(proba_d)
        return probas.set_index(self.df_test.index)

    def predict(self):
        _, truths, predictions, logits, loss_eval = self.evaluate(self.testloader)
        self.loss_eval = loss_eval
        uar, _, _, _, _ = self.evaluate(self.trainloader)
        probas = self.get_probas(logits)
        report = Reporter(
            truths.numpy().astype(float),
            predictions.numpy(),
            self.run,
            self.epoch,
            probas=probas,
            context=self.context,
        )
        try:
            report.result.loss = self.loss
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        report.result.loss_eval = self.loss_eval
        report.result.train = uar
        return report

    def get_predictions(self):
        _, _, predictions, logits, _ = self.evaluate(self.testloader)
        return predictions.numpy(), self.get_probas(logits)

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
