# adms.py
"""
Artifact Detection Modules (ADMs) for deepfake detection.

Adapted for nkululeko's aggregated features (mean-pooled, not time-series).

Modules:
- TimeADM    : Residual MLP for SSL aggregated features
- SpectralADM: Residual MLP for spectral aggregated features
- PhaseADM   : MLP for phase aggregated features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Time-domain Artifact Detector (Residual MLP)
# --------------------------------------------------
class TimeADM(nn.Module):
    """
    Detects temporal artifacts using aggregated SSL features.

    Uses a 3-layer residual MLP with LayerNorm and GELU activations
    for stable training across varying batch sizes.

    Input:
        x : (B, D) - aggregated SSL embeddings (e.g., 768 or 1024 dim)
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # x: (B, D) - aggregated features
        if x.dim() == 3:
            x = x.squeeze(-1)

        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)

        # Residual connection (fc2 preserves hidden_dim)
        residual = x
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + residual

        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc_out(x)
        return x


# --------------------------------------------------
# Spectral Artifact Detector (Residual MLP)
# --------------------------------------------------
class SpectralADM(nn.Module):
    """
    Detects spectral artifacts using aggregated spectral features.

    Uses LazyLinear for the first layer to handle variable input dimensions,
    followed by a residual MLP with LayerNorm and GELU activations.

    Input:
        x : (B, F) - aggregated spectral features (e.g., ~160 dim fbank)
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim=None, hidden_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.LazyLinear(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # x: (B, F) or (B, F, T) - handle both cases
        if x.dim() == 3:
            x = x.squeeze(-1)

        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)

        # Residual connection (fc2 preserves hidden_dim)
        residual = x
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + residual

        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc_out(x)
        return x


# --------------------------------------------------
# Phase Artifact Detector (MLP)
# --------------------------------------------------
class PhaseADM(nn.Module):
    """
    Detects phase artifacts using aggregated phase features.

    Uses a residual MLP matching TimeADM/SpectralADM architecture,
    now that per-bin STFT features provide rich phase information
    (e.g., 514-dim with fft_length=512).

    Input:
        x : (B, D) - aggregated phase features
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # x: (B, D) or (B, T, D) - aggregated features
        if x.dim() == 3:
            x = x.squeeze(1)

        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)

        # Residual connection
        residual = x
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + residual

        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc_out(x)
        return x


# --------------------------------------------------
# Multi-stream Artifact Detection Model
# --------------------------------------------------
class DeepfakeADMModel(nn.Module):
    """
    Multi-stream Artifact Detection Model.

    Fuses scores from time, spectral, and phase ADMs.
    Supports fusion methods: avg, max, weighted, concat, gated.

    Outputs a single artifact score per utterance.
    """

    def __init__(
        self,
        ssl_feat_dim,
        phase_feat_dim,
        fusion="gated",
        branches=None,
        hidden_dim=256,
    ):
        super().__init__()
        if branches is None:
            branches = ["time", "spectral", "phase"]
        self.fusion = fusion
        self.branches = branches

        # Create only requested branches with configurable hidden_dim
        self.time_adm = (
            TimeADM(ssl_feat_dim, hidden_dim=hidden_dim) if "time" in branches else None
        )
        self.spec_adm = (
            SpectralADM(hidden_dim=hidden_dim) if "spectral" in branches else None
        )
        self.phase_adm = (
            PhaseADM(phase_feat_dim, hidden_dim=hidden_dim)
            if "phase" in branches
            else None
        )

        num_branches = len(branches)
        if fusion == "weighted":
            self.weights = nn.Parameter(torch.ones(num_branches))
        elif fusion == "concat":
            self.fusion_fc = nn.Linear(num_branches, 1)
        elif fusion == "gated":
            self.gate_fc = nn.Linear(num_branches, num_branches)

    def forward(self, ssl_feats, spec_feats, phase_feats):
        """
        Args:
            ssl_feats   : (B, D, T)
            spec_feats  : (B, F, T)
            phase_feats : (B, T, Dp)

        Returns:
            artifact_score : (B,)
        """
        # Compute scores only for active branches
        branch_scores = []
        if self.time_adm is not None:
            branch_scores.append(self.time_adm(ssl_feats))
        if self.spec_adm is not None:
            branch_scores.append(self.spec_adm(spec_feats))
        if self.phase_adm is not None:
            branch_scores.append(self.phase_adm(phase_feats))

        scores = torch.cat(branch_scores, dim=1)

        if self.fusion == "avg":
            out = scores.mean(dim=1, keepdim=True)
        elif self.fusion == "max":
            out = scores.max(dim=1, keepdim=True)[0]
        elif self.fusion == "weighted":
            w = F.softmax(self.weights, dim=0)
            out = sum(w[i] * branch_scores[i] for i in range(len(branch_scores)))
        elif self.fusion == "concat":
            out = self.fusion_fc(scores)
        elif self.fusion == "gated":
            gates = torch.sigmoid(self.gate_fc(scores))
            gated_scores = gates * scores
            out = gated_scores.sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        return out.squeeze(1)

    @torch.no_grad()
    def score(self, ssl_feats, spec_feats, phase_feats):
        return self.forward(ssl_feats, spec_feats, phase_feats)
