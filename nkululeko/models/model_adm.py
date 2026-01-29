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
        
        # Binary classification loss (BCE)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.util.debug("using ADM model with BCE loss function")
        
        # Device setup
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        
        # Determine actual feature dimensions from the data
        # Features are: wav2vec2 (768) + sptk (4)
        total_feats = feats_train.shape[1]
        self.ssl_feat_dim = 768  # wav2vec2 dimension
        self.sptk_feat_dim = total_feats - self.ssl_feat_dim  # remaining features
        
        self.util.debug(f"ADM feature split: SSL={self.ssl_feat_dim}, SPTK={self.sptk_feat_dim}")
        
        # ADM architecture parameters
        fusion = self.util.config_val("MODEL", "adm.fusion", "weighted")
        
        # Initialize ADM model with actual feature dimensions
        self.model = DeepfakeADMModel(
            ssl_feat_dim=self.ssl_feat_dim,
            phase_feat_dim=self.sptk_feat_dim,  # Use actual SPTK dimension
            fusion=fusion
        ).to(self.device)
        
        # Learning rate and optimizer
        self.learning_rate = float(
            self.util.config_val("MODEL", "learning_rate", 0.0001)
        )
        optimizer_type = self.util.config_val("MODEL", "optimizer", "adam")
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            self.util.error(f"unknown optimizer: {optimizer_type}")
        
        # Batch size
        self.batch_size = int(self.util.config_val("MODEL", "batch_size", 32))
        self.num_workers = self.n_jobs
        
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
        self.model.train()
        losses = []
        for features, labels in self.trainloader:
            # Convert features to float32 to match model dtype
            features = features.float()
            
            # For binary classification, convert labels to float
            labels_float = labels.float().to(self.device)
            
            # Extract multi-stream features
            # This assumes features are concatenated: [ssl_feats | spec_feats | phase_feats]
            # You may need to adjust based on actual feature structure
            ssl_feats, spec_feats, phase_feats = self._split_features(features)
            
            # Forward pass
            logits = self.model(
                ssl_feats.to(self.device),
                spec_feats.to(self.device),
                phase_feats.to(self.device)
            )
            
            loss = self.criterion(logits, labels_float)
            losses.append(loss.item())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.loss = (np.asarray(losses)).mean()

    def _split_features(self, features):
        """
        Split concatenated features into SSL, spectral, and phase streams.
        
        Features from nkululeko are already aggregated (not time-series):
        - wav2vec2: 768 dimensions (mean pooled SSL embeddings)
        - sptk: variable dimensions (stft/fbank statistics)
        
        For the MLP-based ADM architecture working with aggregated features.
        """
        # Split: wav2vec2 (0:ssl_feat_dim), sptk (ssl_feat_dim:end)
        ssl_feats = features[:, :self.ssl_feat_dim]  # (B, 768)
        sptk_feats = features[:, self.ssl_feat_dim:]  # (B, sptk_feat_dim)
        
        # MLP-based ADMs expect (B, D) - just pass as is
        return ssl_feats, sptk_feats, sptk_feats

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
                    ssl_feats.to(device),
                    spec_feats.to(device),
                    phase_feats.to(device)
                )
                
                logits[start_index:end_index] = batch_logits
                targets[start_index:end_index] = labels
                
                loss = self.criterion(
                    batch_logits.to(device),
                    labels.float().to(device)
                )
                losses.append(loss.item())
        
        self.loss_eval = (np.asarray(losses)).mean()
        
        # Convert logits to predictions (threshold at 0.5)
        predictions = (torch.sigmoid(logits) > 0.5).long()
        
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
            self.util.debug(f"Warning: NaN values detected in probabilities, replacing with 0.5")
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
        
        try:
            report.result.loss = self.loss
        except AttributeError:
            pass
        try:
            report.result.loss_eval = self.loss_eval
        except AttributeError:
            pass
        
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
        
        self.model = DeepfakeADMModel(
            ssl_feat_dim=self.ssl_feat_dim,
            phase_feat_dim=self.sptk_feat_dim,
            fusion=fusion
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(self.store_path))
        except FileNotFoundError:
            self.util.error(f"model file not found: {self.store_path}")
        self.model.eval()

    class MLP:
        """Placeholder for compatibility."""
        pass
            