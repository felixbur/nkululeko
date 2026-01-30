# adms.py
"""
Artifact Detection Modules (ADMs) for deepfake detection - MLP version.

Adapted for nkululeko's aggregated features (mean-pooled, not time-series).

Modules:
- TimeADM    : MLP for SSL aggregated features
- SpectralADM: MLP for spectral aggregated features
- PhaseADM   : MLP for phase aggregated features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Time-domain Artifact Detector (MLP version)
# --------------------------------------------------
class TimeADM(nn.Module):
    """
    Detects temporal artifacts using aggregated SSL features.

    Input:
        x : (B, D) - aggregated SSL embeddings
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, D) - aggregated features
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove time dimension if present

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# --------------------------------------------------
# Spectral Artifact Detector (MLP version)
# --------------------------------------------------
class SpectralADM(nn.Module):
    """
    Detects spectral artifacts using aggregated spectral features.

    Input:
        x : (B, F) - aggregated spectral features
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim=None, hidden_dim=128):
        super().__init__()
        # feat_dim will be determined dynamically from input if not provided
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.fc1 = None  # Lazy initialization
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, F) or (B, F, T) - handle both cases
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove time dimension

        # Lazy initialization on first forward pass
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# --------------------------------------------------
# Phase Artifact Detector (MLP version)
# --------------------------------------------------
class PhaseADM(nn.Module):
    """
    Detects phase artifacts using aggregated phase features.

    Input:
        x : (B, D) - aggregated phase features
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, D) or (B, T, D) - aggregated features
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove time dimension
        elif x.dim() == 2:
            pass  # Already correct shape

        # Simple MLP without problematic normalization
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# --------------------------------------------------
# Multi-stream Artifact Detection Model
# --------------------------------------------------
class DeepfakeADMModel(nn.Module):
    """
    Multi-stream Artifact Detection Model.

    Outputs a single artifact score per utterance.
    """

    def __init__(
        self,
        ssl_feat_dim,
        phase_feat_dim,
        fusion="weighted",
    ):
        super().__init__()
        self.fusion = fusion

        self.time_adm = TimeADM(ssl_feat_dim)
        self.spec_adm = SpectralADM()
        self.phase_adm = PhaseADM(phase_feat_dim)

        if fusion == "weighted":
            self.weights = nn.Parameter(torch.ones(3))
        elif fusion == "concat":
            self.fusion_fc = nn.Linear(3, 1)

    def forward(self, ssl_feats, spec_feats, phase_feats):
        """
        Args:
            ssl_feats   : (B, D, T)
            spec_feats  : (B, F, T)
            phase_feats : (B, T, Dp)

        Returns:
            artifact_score : (B,)
        """
        t = self.time_adm(ssl_feats)
        s = self.spec_adm(spec_feats)
        p = self.phase_adm(phase_feats)

        scores = torch.cat([t, s, p], dim=1)

        if self.fusion == "avg":
            out = scores.mean(dim=1, keepdim=True)
        elif self.fusion == "max":
            out = scores.max(dim=1, keepdim=True)[0]
        elif self.fusion == "weighted":
            w = F.softmax(self.weights, dim=0)
            out = w[0] * t + w[1] * s + w[2] * p
        elif self.fusion == "concat":
            out = self.fusion_fc(scores)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        return out.squeeze(1)

    @torch.no_grad()
    def score(self, ssl_feats, spec_feats, phase_feats):
        return self.forward(ssl_feats, spec_feats, phase_feats)
