# model_adm.py
"""
Artifact Detection Module (ADM) model for deepfake detection.

Multi-stream neural model that detects synthesis artifacts using:
- TimeADM: temporal/micro-prosodic artifacts from SSL features
- SpectralADM: spectral/vocoder artifacts from mel-filterbank features
- PhaseADM: phase-dynamics artifacts from STFT features
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score

import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model
from nkululeko.models.model_adm_core import DeepfakeADMModel
from nkululeko.optimizers import (
    get_optimizer,
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)
from nkululeko.reporting.reporter import Reporter


class ADMModel(Model):
    """ADM = Artifact Detection Module for deepfake detection."""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor, taking all dataframes.

        Args:
            df_train (pd.DataFrame): The train labels.
            df_test (pd.DataFrame): The test labels.
            feats_train (pd.DataFrame): The train features.
            feats_test (pd.DataFrame): The test features.
        """
        super().__init__(df_train, df_test, feats_train, feats_test)
        super().set_model_type("ann")
        self.name = "adm"
        self.target = glob_conf.config["DATA"]["target"]

        # Manual seed
        manual_seed = eval(self.util.config_val("MODEL", "random_seed", "False"))
        if manual_seed:
            self.util.debug(f"seeding random to {manual_seed}")
            torch.manual_seed(int(manual_seed))

        # Get labels and class number
        labels = glob_conf.labels
        self.class_num = len(labels)

        # Device setup (needed before loss initialization)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)

        # Loss function selection
        loss_type = self.util.config_val("MODEL", "loss", "bce")

        def _parse_float(val, default):
            """Parse float values that may include inline comments."""
            try:
                return float(str(val).split("#")[0].strip())
            except (TypeError, ValueError):
                return default

        if loss_type == "focal":
            from nkululeko.losses.loss_focal import FocalLoss

            alpha = _parse_float(
                self.util.config_val("MODEL", "focal.alpha", 0.25), 0.25
            )
            gamma = _parse_float(self.util.config_val("MODEL", "focal.gamma", 2.0), 2.0)
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
            self.util.debug(
                f"using ADM model with Focal Loss (alpha={alpha}, gamma={gamma})"
            )

        elif loss_type == "weighted_bce":
            from nkululeko.losses.loss_bce import (
                WeightedBCEWithLogitsLoss,
                compute_class_weights,
            )

            # Compute class weights from training data
            class_weight_mode = self.util.config_val("MODEL", "class_weight", "auto")
            if class_weight_mode == "auto":
                pos_weight = compute_class_weights(
                    df_train[self.target], labels, mode="balanced"
                )
                self.util.debug(f"auto class weight: pos_weight={pos_weight:.3f}")
            else:
                pos_weight = _parse_float(class_weight_mode, 1.0)
                self.util.debug(f"manual class weight: pos_weight={pos_weight}")
            self.criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
            self.util.debug("using ADM model with Weighted BCE loss function")

        else:  # default: bce
            from nkululeko.losses.loss_bce import BCEWithLogitsLoss

            self.criterion = BCEWithLogitsLoss()
            self.util.debug("using ADM model with BCE loss function")

        # Determine actual feature dimensions from the data
        # Features are concatenated: wav2vec2 (integer cols) + SPTK (named cols)
        total_feats = feats_train.shape[1]
        # Auto-detect SSL dim from integer-named columns
        ssl_cols = [c for c in feats_train.columns if isinstance(c, (int, np.integer))]
        self.ssl_feat_dim = len(ssl_cols) if ssl_cols else 768
        self.sptk_feat_dim = total_feats - self.ssl_feat_dim

        # Analyze SPTK column names to split spectral vs phase features
        sptk_cols = [c for c in feats_train.columns if isinstance(c, str)]
        self.fbank_indices = []
        self.stft_indices = []
        for i, c in enumerate(sptk_cols):
            col_idx = self.ssl_feat_dim + i
            if "fbank" in str(c):
                self.fbank_indices.append(col_idx)
            elif "stft" in str(c):
                self.stft_indices.append(col_idx)
            else:
                self.fbank_indices.append(col_idx)  # default to spectral

        self.util.debug(
            f"ADM feature split: SSL={self.ssl_feat_dim}, "
            f"spectral(fbank)={len(self.fbank_indices)}, "
            f"phase(stft)={len(self.stft_indices)}"
        )

        # ADM architecture parameters
        fusion = self.util.config_val("MODEL", "adm.fusion", "weighted")
        phase_dim = len(self.stft_indices) if self.stft_indices else self.sptk_feat_dim

        # Initialize ADM model with actual feature dimensions
        self.model = DeepfakeADMModel(
            ssl_feat_dim=self.ssl_feat_dim,
            phase_feat_dim=phase_dim,
            fusion=fusion,
        ).to(self.device)

        # Learning rate and optimizer
        self.optimizer, self.learning_rate = get_optimizer(
            self.model.parameters(),
            self.util,
            default_lr=0.0001,
            default_optimizer="adamw",
        )

        # Learning rate scheduler
        self.scheduler, self.scheduler_type, self.scheduler_needs_init = get_scheduler(
            self.optimizer, self.util, default_scheduler="cosine"
        )

        # Batch size
        self.batch_size = int(self.util.config_val("MODEL", "batch_size", 32))
        self.num_workers = self.n_jobs

        # Training hyperparameters (read once to avoid repeated logging)
        self.max_grad_norm = float(
            self.util.config_val("MODEL", "max_grad_norm", "0.0")
        )
        self.feature_noise = float(
            self.util.config_val("MODEL", "feature_noise", "0.0")
        )
        self.threshold = float(self.util.config_val("MODEL", "threshold", "0.5"))

        # Handle NaN values
        if feats_train.isna().to_numpy().any():
            self.util.debug(
                f"Model, train: replacing {feats_train.isna().sum().sum()} NANs with 0"
            )
            feats_train = feats_train.fillna(0)
        if feats_test.isna().to_numpy().any():
            self.util.debug(
                f"Model, test: replacing {feats_test.isna().sum().sum()} NANs with 0"
            )
            feats_test = feats_test.fillna(0)

        # Set up data loaders
        self.trainloader = self.get_loader(feats_train, df_train, True)
        self.testloader = self.get_loader(feats_test, df_test, False)

    def set_testdata(self, data_df, feats_df):
        self.df_test = data_df
        self.feats_test = feats_df
        self.testloader = self.get_loader(feats_df, data_df, False)

    def reset_test(self, df_test, feats_test):
        self.testloader = self.get_loader(feats_test, df_test, False)
        self.df_test = df_test
        self.feats_test = feats_test

    def train(self):
        """Train the ADM model for one epoch."""
        # Initialize cosine scheduler on first call (needs total steps)
        if self.scheduler_needs_init and self.scheduler is None:
            self.scheduler = initialize_cosine_scheduler(
                self.optimizer, self.util, steps_per_epoch=len(self.trainloader)
            )
            self.scheduler_needs_init = False

        self.model.train()
        losses = []
        for features, labels in self.trainloader:
            features = features.float()
            labels_float = labels.float().to(self.device)

            # Feature-level Gaussian noise augmentation for regularization
            if self.feature_noise > 0:
                noise = torch.randn_like(features) * self.feature_noise
                features = features + noise

            ssl_feats, spec_feats, phase_feats = self._split_features(features)

            logits = self.model(
                ssl_feats.to(self.device),
                spec_feats.to(self.device),
                phase_feats.to(self.device),
            )

            loss = self.criterion(logits, labels_float)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
            self.optimizer.step()

            # Step scheduler per batch (for cosine)
            step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=True)

        # Step scheduler per epoch (for step/exponential)
        step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=False)

        self.loss = (np.asarray(losses)).mean()

    def _split_features(self, features):
        """Split concatenated features into SSL, spectral, and phase streams.

        Features are concatenated as: wav2vec2 (first ssl_feat_dim cols) +
        SPTK features split by column name (fbank -> spectral, stft -> phase).
        """
        ssl_feats = features[:, : self.ssl_feat_dim]
        if self.fbank_indices:
            spec_feats = features[:, self.fbank_indices]
        else:
            spec_feats = features[:, self.ssl_feat_dim :]
        if self.stft_indices:
            phase_feats = features[:, self.stft_indices]
        else:
            phase_feats = features[:, self.ssl_feat_dim :]
        return ssl_feats, spec_feats, phase_feats

    def evaluate(self, model, loader, device):
        """Evaluate the ADM model."""
        logits = torch.zeros(len(loader.dataset))
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        losses = []

        with torch.no_grad():
            for index, (features, labels) in enumerate(loader):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)

                # Convert features to float32
                features = features.float()

                # Split features
                ssl_feats, spec_feats, phase_feats = self._split_features(features)

                # Forward pass
                batch_logits = model(
                    ssl_feats.to(device), spec_feats.to(device), phase_feats.to(device)
                )

                logits[start_index:end_index] = batch_logits
                targets[start_index:end_index] = labels

                loss = self.criterion(
                    batch_logits.to(device), labels.float().to(device)
                )
                losses.append(loss.item())

        self.loss_eval = (np.asarray(losses)).mean()

        predictions = (torch.sigmoid(logits) > self.threshold).long()

        # Calculate UAR (macro recall)
        uar = recall_score(targets.numpy(), predictions.numpy(), average="macro")

        return uar, targets, predictions, logits

    def get_probas(self, logits):
        """Convert logits to probability dataframe."""
        # For binary classification with BCEWithLogitsLoss
        proba_d = {}
        classes = self.df_test[self.target].unique()
        classes.sort()

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).numpy()

        # Handle NaN values
        if np.isnan(probs).any():
            self.util.debug(
                "Warning: NaN values detected in probabilities, replacing with 0.5"
            )
            probs = np.nan_to_num(probs, nan=0.5)

        # Assuming binary: class 0 (real) and class 1 (fake)
        if len(classes) == 2:
            proba_d[classes[0]] = 1 - probs
            proba_d[classes[1]] = probs
        else:
            # Handle multi-class case (shouldn't happen with binary BCE)
            for i, cls in enumerate(classes):
                proba_d[cls] = probs if i == 1 else 1 - probs

        probas = pd.DataFrame(proba_d)
        probas = probas.set_index(self.df_test.index)
        return probas

    def predict(self):
        """Make predictions on test data."""
        _, truths, predictions, logits = self.evaluate(
            self.model, self.testloader, self.device
        )
        uar, _, _, _ = self.evaluate(self.model, self.trainloader, self.device)
        probas = self.get_probas(logits)
        report = Reporter(truths, predictions, self.run, self.epoch, probas=probas)

        if hasattr(self, "loss"):
            report.result.loss = self.loss
        if hasattr(self, "loss_eval"):
            report.result.loss_eval = self.loss_eval

        report.result.train = uar
        return report

    def get_predictions(self):
        """Get predictions without creating a full report."""
        _, _, predictions, logits = self.evaluate(
            self.model, self.testloader, self.device
        )
        return (predictions.numpy(), self.get_probas(logits))

    def get_loader(self, df_x, df_y, shuffle):
        """Create a data loader for training or testing."""
        data = []
        for i in range(len(df_x)):
            data.append([df_x.values[i], df_y[self.target].iloc[i]])
        return torch.utils.data.DataLoader(
            data, shuffle=shuffle, batch_size=self.batch_size
        )

    def store(self):
        """Store the model to disk."""
        torch.save(self.model.state_dict(), self.store_path)

    def load(self, run, epoch):
        """Load a trained model from disk."""
        self.set_id(run, epoch)
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.store_path = dir + name

        # Recreate model architecture with actual dimensions
        fusion = self.util.config_val("MODEL", "adm.fusion", "weighted")
        phase_dim = len(self.stft_indices) if self.stft_indices else self.sptk_feat_dim

        self.model = DeepfakeADMModel(
            ssl_feat_dim=self.ssl_feat_dim,
            phase_feat_dim=phase_dim,
            fusion=fusion,
        ).to(self.device)

        try:
            self.model.load_state_dict(torch.load(self.store_path))
        except FileNotFoundError:
            self.util.error(f"model file not found: {self.store_path}")
        self.model.eval()
