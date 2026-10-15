"""LSTM_Reg_model: [MODEL] type = lstm_reg.

Regression sibling of model_lstm.py (GH #256), same relationship as
model_mlp.py/model_mlp_regression.py: same raw-audio-bypass architecture
(FEATS.type = [] required), duplicated Dataset/net classes (this repo's
established convention for a model family's classification/regression
split, confirmed by model_mlp_regression.py fully duplicating model_mlp.py's
MLP/Dataset rather than importing them) with a 1-output regression head
and MSE/MAE/CCC/PCC loss instead of cross-entropy.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from audmetric import (
    concordance_cc,
    mean_absolute_error,
    mean_squared_error,
    pearson_cc,
)
from torch.utils.data import DataLoader, Dataset

from nkululeko.losses.loss_ccc import ConcordanceCorCoeff
from nkululeko.losses.loss_pcc import PearsonCorCoeff
from nkululeko.models.model import Model
from nkululeko.models.model_lstm import _TARGET_SR, _pad_or_tile
from nkululeko.models.rnn_config import RNNConfig
from nkululeko.optimizers import (
    get_optimizer,
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.random_seed import parse_seed


class _FrameSequenceDataset(Dataset):
    """Duplicated from model_lstm.py's class of the same name (see module
    docstring) -- identical raw-audio-to-frame-sequence pipeline, the
    label is just read as a plain value either way (cast to float only in
    train()/evaluate() here, vs. long() there)."""

    def __init__(self, df, target, cfg, domain_labels=None):
        self.df = df
        self.target = target
        self.cfg = cfg
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
        if self.mel is not None:
            signal = torch.from_numpy(waveform)
            spec = self.db(self.mel(signal))
            return spec.transpose(0, 1).float()
        result = self.smile.process_signal(waveform, _TARGET_SR)
        return torch.from_numpy(result.to_numpy()).float()

    def __getitem__(self, idx):
        import audiofile
        import audresample
        import warnings

        file, start, end = self.df.index[idx]
        if pd.isna(end):
            signal, sr = audiofile.read(file, offset=start, always_2d=True)
        else:
            signal, sr = audiofile.read(
                file, duration=end - start, offset=start, always_2d=True
            )
        signal = np.asarray(signal).mean(axis=0)
        if sr != _TARGET_SR:
            signal = audresample.resample(signal, sr, _TARGET_SR).squeeze()

        if signal.shape[0] == 0:
            warnings.warn(
                f"lstm_reg: no audio read from {file} (start={start}, end={end}); "
                "using silence for this sample",
                stacklevel=2,
            )
        signal = _pad_or_tile(signal, self.cfg.max_len).astype(np.float32)
        frames = self._frames(signal)

        row = self.df.iloc[idx]
        label = np.float32(row[self.target])
        if self.domain_labels is None:
            return frames, label
        return frames, label, torch.from_numpy(self.domain_labels[idx])


class _RNNNet(nn.Module):
    """Duplicated from model_lstm.py's class of the same name, with a
    1-output regression head instead of an num_classes-way classifier."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
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
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.feat_dim = hidden_size * (2 if bidirectional else 1)
        self.out = nn.Linear(self.feat_dim, 1)

    def forward(self, x, return_features=False):
        outputs, _ = self.rnn(x)
        pooled = outputs.mean(dim=1)
        value = self.out(pooled)
        if return_features:
            return value, pooled
        return value


class LSTM_Reg_model(Model):
    """LSTM regression = frame-sequence recurrent network for a continuous target."""

    is_classifier = False

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("ann")
        self.name = "lstm_reg"
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

        criterion = self.util.config_val("MODEL", "loss", "mse")
        if criterion == "mse":
            self.criterion = torch.nn.MSELoss()
        elif criterion == "mae":
            self.criterion = torch.nn.L1Loss()
        elif criterion == "1-ccc":
            self.criterion = ConcordanceCorCoeff()
        elif criterion == "1-pcc":
            self.criterion = PearsonCorCoeff()
        else:
            self.util.error(f"unknown loss function: {criterion}")
        self.util.debug(f"training model with {criterion} loss function")

        self.cfg = RNNConfig.from_util(self.util)
        self.device = self.cfg.device
        self.util.debug(
            f"lstm_reg: feature_source={self.cfg.feature_source}, "
            f"hidden_size={self.cfg.hidden_size}, num_layers={self.cfg.num_layers}"
        )

        self.trainloader = self.get_loader(df_train, shuffle=True, train=True)
        self.testloader = self.get_loader(df_test, shuffle=False)

        input_size = self.trainloader.dataset[0][0].shape[-1]
        self.net = _RNNNet(
            input_size,
            self.cfg.hidden_size,
            self.cfg.num_layers,
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
            labels = labels.to(self.device)

            loss = self._optimize(
                self.optimizer,
                lambda: self._forward_loss(frames, labels, domain_labels),
                skip_non_finite=True,  # ccc/pcc can be non-finite on a one-value batch
                modules=[self.net],
            )

            if loss is not None:
                losses.append(loss.item())
                step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=True)

        step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=False)
        self.loss = float(np.mean(losses)) if losses else 0.0

    def _forward_loss(self, frames, labels, domain_labels):
        if self.dann_heads is None:
            return self.criterion(self.net(frames).reshape(-1), labels)
        value, hidden = self.net(frames, return_features=True)
        return self.criterion(value.reshape(-1), labels) + self.dann_heads.loss(
            hidden, domain_labels
        )

    def evaluate(self, loader):
        logits = torch.zeros(len(loader.dataset))
        targets = torch.zeros(len(loader.dataset))
        self.net.eval()
        losses = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                frames, labels = batch[0], batch[1]
                start_index = index * loader.batch_size
                end_index = min((index + 1) * loader.batch_size, len(loader.dataset))
                logits[start_index:end_index] = self.net(
                    frames.to(self.device)
                ).reshape(-1)
                targets[start_index:end_index] = labels
                loss = self.criterion(
                    logits[start_index:end_index].to(self.device),
                    labels.to(self.device),
                )
                # ccc/pcc are non-finite on a one-value batch (e.g. the last
                # one); leave it out of the loss, keep its predictions.
                if torch.isfinite(loss):
                    losses.append(loss.item())

        loss_eval = float(np.mean(losses)) if losses else 0.0
        predictions = logits
        measure = self.util.config_val("MODEL", "measure", "mse")
        if measure == "mse":
            result = mean_squared_error(targets.numpy(), predictions.numpy())
        elif measure == "mae":
            result = mean_absolute_error(targets.numpy(), predictions.numpy())
        elif measure == "ccc":
            result = concordance_cc(targets.numpy(), predictions.numpy())
        elif measure == "pcc":
            result = pearson_cc(targets.numpy(), predictions.numpy())
        else:
            self.util.error(f"unknown measure: {measure}")
        return result, targets, predictions, loss_eval

    def predict(self):
        _, truths, predictions, loss_eval = self.evaluate(self.testloader)
        self.loss_eval = loss_eval
        result, _, _, _ = self.evaluate(self.trainloader)
        report = Reporter(
            truths.numpy(),
            predictions.numpy(),
            self.run,
            self.epoch,
            context=self.context,
        )
        try:
            report.result.loss = self.loss
        except AttributeError:
            pass
        report.result.loss_eval = self.loss_eval
        report.result.train = result
        return report

    def get_predictions(self):
        _, _, predictions, _ = self.evaluate(self.testloader)
        return predictions.numpy(), None

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
