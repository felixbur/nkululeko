"""AasistModel: [MODEL] type = aasist.

Waveform-level end-to-end AASIST (SSL frontend + spectro-temporal graph
attention backend, see model_aasist_core.py). Unlike models that work on
precomputed features, it reads raw audio directly from df_train/df_test
(via audiofile.read(), same pattern as model_tuned.py) and bypasses
nkululeko's FeatureExtractor entirely, so a config using this model must
set FEATS.type = [] (no features to extract). model_type is set to "ann" (not "finetuned"): unlike
TunedModel's HuggingFace Trainer, AASIST's train() runs exactly one
epoch per call, so modelrunner.do_epochs()'s own per-epoch loop (and
therefore MODEL.patience-based early stopping, EXP.traindevtest's
dev/test split, and every other existing epoch-level machinery) applies
identically to how it does for model_adm.py -- the model this is meant
to be directly comparable against.
"""

import itertools
import warnings

import audresample
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from nkululeko.models.aasist_config import AasistConfig
from nkululeko.models.domain_adversarial import DomainAdversarialHead
from nkululeko.models.model import Model
from nkululeko.models.model_aasist_core import AasistBackend
from nkululeko.optimizers import (
    get_optimizer,
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.random_seed import parse_seed

# The SSL frontends (wav2vec2/XLS-R) expect 16 kHz mono input.
_TARGET_SR = 16000


def _pad_or_tile(signal, max_len):
    """Pad-by-tiling (upstream's convention) or truncate to a fixed length.

    An empty signal (e.g. a zero-duration segment) cannot be tiled; it is
    replaced by silence of max_len samples.
    """
    sig_len = signal.shape[0]
    if sig_len == 0:
        return np.zeros(max_len, dtype=signal.dtype)
    if sig_len >= max_len:
        return signal[:max_len]
    num_repeats = int(max_len / sig_len) + 1
    return np.tile(signal, num_repeats)[:max_len]


class _WaveformDataset(Dataset):
    """Reads raw waveforms directly from disk, bypassing FeatureExtractor.

    Mirrors model_tuned.py's own raw-audio loading (audiofile.read() off
    the (file, start, end) segmented index), including its handling of
    whole-file (non-segmented) rows, except that it uses `pd.isna(end)`
    rather than `end == pd.NaT`, which is always False. Unlike model_tuned.py
    it also downmixes multi-channel audio to mono and resamples to 16 kHz,
    since the SSL frontend only works on 16 kHz mono input.
    """

    def __init__(self, df, target, cfg, dann_label_maps=None):
        self.df = df
        self.target = target
        self.cfg = cfg
        # {column: {raw_value: int_index}} -- when set, items are
        # (waveform, label, domain_labels) instead of (waveform, label).
        # Only the train loader sets it.
        self.dann_label_maps = dann_label_maps

    def __len__(self):
        return len(self.df)

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
                f"aasist: no audio read from {file} (start={start}, end={end}); "
                "using silence for this sample",
                stacklevel=2,
            )
        signal = _pad_or_tile(signal, self.cfg.max_len)
        row = self.df.iloc[idx]
        label = row[self.target]
        waveform = torch.tensor(signal, dtype=torch.float32)
        if self.dann_label_maps is None:
            return waveform, label
        domain_labels = torch.tensor(
            [self.dann_label_maps[col][row[col]] for col in self.cfg.dann_columns],
            dtype=torch.long,
        )
        return waveform, label, domain_labels


class AasistModel(Model):
    """AASIST = spectro-temporal graph attention network for deepfake detection."""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("ann")
        self.name = "aasist"
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
        if self.class_num != 2:
            self.util.error(
                f"aasist model requires exactly 2 classes (got {self.class_num}: "
                f"{labels}) -- its output layer is a fixed binary (real/fake) head"
            )

        self.cfg = AasistConfig.from_util(self.util)
        self.device = self.cfg.device
        self.util.debug(
            f"aasist: SSL frontend {self.cfg.ssl_model}, max_len={self.cfg.max_len}"
        )
        self.net = AasistBackend(
            self.cfg.ssl_model,
            layer_pooling=self.cfg.ssl_layer_pooling,
            freeze_ssl=self.cfg.freeze_ssl_frontend,
        ).to(self.device)

        self._build_criterion(df_train)
        self._build_dann_heads(df_train)

        params = (
            itertools.chain(self.net.parameters(), self.dann_heads.parameters())
            if self.dann_heads is not None
            else self.net.parameters()
        )
        self.optimizer, self.learning_rate = get_optimizer(
            params, self.util, default_lr=1e-5, default_optimizer="adam"
        )
        self.scheduler, self.scheduler_type, self.scheduler_needs_init = get_scheduler(
            self.optimizer, self.util, default_scheduler="none"
        )

        self.trainloader = self.get_loader(df_train, shuffle=True, train=True)
        self.testloader = self.get_loader(df_test, shuffle=False)

    def _build_dann_heads(self, df_train):
        """One DomainAdversarialHead per MODEL.dann_columns entry, attached
        to the backend's pooled feature vector. Label maps come from
        df_train; the heads only ever run on the train split."""
        self.dann_label_maps = {}
        self.dann_heads = None
        if not self.cfg.dann_columns:
            return
        heads = {}
        for col in self.cfg.dann_columns:
            if col not in df_train.columns:
                self.util.error(
                    f"MODEL.dann_columns includes '{col}', which is not a "
                    "column of the training data"
                )
            if df_train[col].isna().any():
                self.util.error(
                    f"MODEL.dann_columns column '{col}' has missing values "
                    "in the training data"
                )
            values = sorted(df_train[col].unique().tolist())
            if len(values) < 2:
                self.util.error(
                    f"MODEL.dann_columns includes '{col}', but the training "
                    f"data has {len(values)} distinct value(s) for it; DANN "
                    "needs at least 2 to discriminate between"
                )
            self.dann_label_maps[col] = {v: i for i, v in enumerate(values)}
            heads[col] = DomainAdversarialHead(
                feat_dim=self.net.feat_dim,
                num_classes=len(values),
                reverse=self.cfg.dann_reverse,
                lambda_=self.cfg.dann_lambda,
            )
        self.dann_heads = nn.ModuleDict(heads).to(self.device)
        self.util.debug(
            f"aasist: DANN heads for {self.cfg.dann_columns} "
            f"(reverse={self.cfg.dann_reverse}, lambda={self.cfg.dann_lambda}, "
            f"weight={self.cfg.dann_weight})"
        )

    def _build_criterion(self, df_train):
        """CrossEntropyLoss over the fixed 2-way output, with optional
        MODEL.class_weight (True = balanced class weights). Only
        MODEL.loss=cross is supported.
        """
        loss_type = self.util.config_val("MODEL", "loss", "cross")
        if loss_type != "cross":
            self.util.error(
                f"aasist model currently only supports MODEL.loss=cross (got '{loss_type}')"
            )
        label_smoothing = self._get_label_smoothing()
        weight = None
        if self.util.config_val_bool("MODEL", "class_weight", False):
            classes = np.arange(self.class_num)
            present = set(df_train[self.target].unique())
            if not set(classes) <= present:
                self.util.error(
                    "MODEL.class_weight needs both classes in the training "
                    f"split, but it only contains {sorted(present)}"
                )
            cw = compute_class_weight(
                class_weight="balanced", classes=classes, y=df_train[self.target]
            )
            weight = torch.tensor(cw, dtype=torch.float32, device=self.device)
            self.util.debug(f"aasist: class weights {cw}")
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )

    def get_loader(self, df, shuffle, train=False):
        dataset = _WaveformDataset(
            df,
            self.target,
            self.cfg,
            dann_label_maps=self.dann_label_maps if train else None,
        )
        # Each __getitem__ does its own audiofile.read() -- CPU-bound work
        # that a single-process loader (num_workers=0) serializes with GPU
        # compute. MODEL.n_jobs (already read by the base Model class into
        # self.n_jobs) parallelizes it; n_jobs=0 keeps the single-process
        # behavior. persistent_workers avoids respawning the worker pool
        # every epoch when num_workers > 0.
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
        epoch and owns the epoch loop/patience, matching model_adm.py)."""
        if self.scheduler_needs_init and self.scheduler is None:
            self.scheduler = initialize_cosine_scheduler(
                self.optimizer, self.util, steps_per_epoch=len(self.trainloader)
            )
            self.scheduler_needs_init = False

        self.net.train()
        losses = []
        for batch in self.trainloader:
            if self.dann_heads is not None:
                waveforms, labels, domain_labels = batch
                domain_labels = domain_labels.to(self.device)
            else:
                waveforms, labels = batch
                domain_labels = None
            waveforms = waveforms.to(self.device)
            labels = labels.long().to(self.device)

            loss = self._forward_loss(waveforms, labels, domain_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=True)
            losses.append(loss.item())

        step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=False)
        self.loss = float(np.mean(losses)) if losses else 0.0

    def _forward_loss(self, waveforms, labels, domain_labels):
        """Task loss, plus one weighted domain-classification loss per
        DANN head when MODEL.dann_columns is set. The heads sit behind a
        gradient-reversal layer (unless dann_reverse = False), so
        minimizing the sum trains the shared features to be useless for
        guessing the domain."""
        if self.dann_heads is None:
            return self.criterion(self.net(waveforms), labels)
        logits, feats = self.net(waveforms, return_features=True)
        loss = self.criterion(logits, labels)
        for i, col in enumerate(self.cfg.dann_columns):
            domain_logits = self.dann_heads[col](feats)
            loss = loss + self.cfg.dann_weight * nn.functional.cross_entropy(
                domain_logits, domain_labels[:, i]
            )
        return loss

    def evaluate(self, loader):
        self.net.eval()
        all_logits, all_targets, losses = [], [], []
        with torch.no_grad():
            for waveforms, labels in loader:
                waveforms = waveforms.to(self.device)
                labels_t = labels.long().to(self.device)
                logits = self.net(waveforms)
                losses.append(self.criterion(logits, labels_t).item())
                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())

        logits = torch.cat(all_logits) if all_logits else torch.empty(0, 2)
        targets = torch.cat(all_targets) if all_targets else torch.empty(0)
        predictions = torch.argmax(logits, dim=1)
        uar = (
            recall_score(targets.numpy(), predictions.numpy(), average="macro")
            if len(targets)
            else 0.0
        )
        return (
            uar,
            targets,
            predictions,
            logits,
            float(np.mean(losses)) if losses else 0.0,
        )

    def get_probas(self, logits):
        probs = torch.softmax(logits, dim=1).numpy()
        proba_d = {c: probs[:, i] for i, c in enumerate(np.arange(self.class_num))}
        probas = pd.DataFrame(proba_d)
        return probas.set_index(self.df_test.index)

    def predict(self):
        """Predict on the current test set.

        Unlike model_adm.py's predict(), this does NOT also evaluate over
        the full trainloader to report a "train" UAR figure: for a
        300M-parameter SSL-frontend model, a second full forward pass over
        every training utterance every single epoch would roughly double
        AASIST's already much higher per-epoch cost compared to ADM's
        cached-feature training, for a purely diagnostic number.
        """
        _, truths, predictions, logits, loss_eval = self.evaluate(self.testloader)
        self.loss_eval = loss_eval
        probas = self.get_probas(logits)
        report = Reporter(
            truths.numpy().astype(float),
            predictions.numpy(),
            self.run,
            self.epoch,
            probas=probas,
            context=self.context,
        )
        if hasattr(self, "loss"):
            report.result.loss = self.loss
        report.result.loss_eval = self.loss_eval
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
