"""AASIST backend architecture, ported from:

Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung,
Bong-Jin Lee, Ha-Jin Yu, Nicholas Evans. "AASIST: Audio Anti-Spoofing
Using Integrated Spectro-Temporal Graph Attention Networks." ICASSP 2022.

via the reference implementation in TakHemlata/SSL_Anti-spoofing
(model.py's GraphAttentionLayer / HtrgGraphAttentionLayer / GraphPool /
Residual_block / Model classes), itself based on clovaai/aasist. Both
repositories are MIT licensed. Ported near-verbatim (renamed for style, no
architectural changes) EXCEPT the SSL frontend: upstream loads XLS-R-300M via fairseq's
checkpoint_utils, which pins an old torch + a fairseq install this
project's environment doesn't carry. HFWav2Vec2Frontend below swaps in
HuggingFace's Wav2Vec2Model (already used elsewhere in nkululeko, e.g.
feat_extract/feats_wav2vec2.py and models/model_tuned.py) for the exact
same architecture family (wav2vec2/XLS-R), so the AASIST *backend* below
follows upstream except for two deliberate fixes (marked in
ResidualBlock.forward and AasistBackend.forward) and the frontend's loading
mechanism.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class HFWav2Vec2Frontend(nn.Module):
    """SSL frontend for AasistBackend, matching upstream's SSLModel.extract_feat()
    interface (a callable returning (batch, frames, hidden) framewise
    embeddings) but loading via HuggingFace instead of fairseq.

    out_dim is read from the loaded checkpoint's own config (e.g. 1024 for
    XLS-R-300M, 768 for wav2vec2-base), rather than hardcoded,
    so a different --ssl_model config value can't silently mismatch the
    backend's first linear layer.

    layer_pooling: "last" (default -- only the final encoder layer's
    hidden states) or
    "weighted" -- a learnable, softmax-normalized scalar per hidden-state
    layer (the CNN feature-extractor's output plus every transformer
    layer, num_hidden_layers + 1 of them) combines all of them, the same
    "weighted sum of hidden states" technique nkululeko's own
    [FINETUNE] layer_pooling=weighted already offers for TunedModel.
    Motivation: for spoof detection, useful signal in wav2vec2/XLS-R can
    sit in lower/middle layers rather than only the last one.

    freeze: if True, the frontend's own parameters are excluded from
    training (requires_grad_(False)) -- PyTorch's autograd then skips
    building a backward graph through the frontend entirely (its output
    has requires_grad=False whenever every parameter feeding it does),
    not just skipping the weight update, which is the actual source of
    the "3-5x faster" speedup layer-selection literature reports for a
    frozen SSL frontend, not merely skipping optimizer.step() on it.
    """

    def __init__(
        self, pretrained_model: str, layer_pooling: str = "last", freeze: bool = False
    ):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model)
        self.out_dim = self.model.config.hidden_size
        self.layer_pooling = layer_pooling
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
        if layer_pooling == "weighted":
            # LayerDrop (config.layerdrop, default 0.1 for XLS-R) randomly
            # skips whole encoder layers during training and does not
            # append a hidden_states entry for a skipped layer -- so
            # len(hidden_states) varies call to call under the default
            # config instead of staying fixed at num_hidden_layers + 1,
            # breaking layer_weights' fixed-size combination (confirmed:
            # 22-24 elements instead of the expected 25 across repeated
            # forward calls in train mode). Disabled here, only for this
            # pooling mode -- "last" pooling never reads hidden_states, so
            # it's unaffected regardless of layerdrop.
            self.model.config.layerdrop = 0.0
            num_layers = self.model.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        if self.layer_pooling == "weighted":
            hidden_states = self.model(
                input_data, output_hidden_states=True
            ).hidden_states
            stacked = torch.stack(hidden_states, dim=0)  # (L, B, T, H)
            weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            return (stacked * weights).sum(dim=0)
        return self.model(input_data).last_hidden_state


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        """x: (#bs, #node, #dim)"""
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_bn(x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_bn(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        """x1, x2: (#bs, #node, #dim)"""
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)

        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_bn(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )
        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_bn(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        z = self.drop(h)
        scores = self.sigmoid(self.proj(z))
        return self._top_k_graph(scores, h, self.k)

    def _top_k_graph(self, scores, h, k):
        """scores: (#bs, #node, 1), h: (#bs, #node, #dim), k: keep ratio."""
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1
        )
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(
                nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1), stride=1
            )

    def forward(self, x):
        identity = x
        # Deliberate deviation from upstream SSL_Anti-spoofing, which computes
        # the BN+SELU pre-activation here but then feeds the raw `x` to conv1
        # (silently discarding it for every non-first block). We apply it, as
        # the standard pre-activation residual block intends. Pretrained
        # upstream AASIST weights are not loaded, so this costs no parity.
        out = self.selu(self.bn1(x)) if not self.first else x
        out = self.conv1(out)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


# Default AASIST hyperparameters, matching upstream's conf.py fallback
# (used whenever conf.py isn't importable there); not user-configurable
# here since the architecture (as opposed to training hyperparameters) is
# what the AASIST paper specifies, not something this port re-tunes.
_FILTS = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
_GAT_DIMS = [64, 32]
_POOL_RATIOS = [0.5, 0.5, 0.5, 0.5]
_TEMPERATURES = [2.0, 2.0, 100.0, 100.0]


class AasistBackend(nn.Module):
    """SSL frontend + AASIST spectro-temporal graph-attention backend.

    forward(x) takes a raw waveform batch (#bs, #samples) or (#bs,
    #samples, 1) and returns (#bs, 2) logits (index order matches
    whatever integer encoding AasistModel's label encoder assigns -- this
    class has no opinion on which class means "real" vs "fake").
    """

    def __init__(
        self, ssl_model: str, layer_pooling: str = "last", freeze_ssl: bool = False
    ):
        super().__init__()
        self.ssl_model = HFWav2Vec2Frontend(
            ssl_model, layer_pooling=layer_pooling, freeze=freeze_ssl
        )
        self.ll = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(ResidualBlock(nb_filts=_FILTS[1], first=True)),
            nn.Sequential(ResidualBlock(nb_filts=_FILTS[2])),
            nn.Sequential(ResidualBlock(nb_filts=_FILTS[3])),
            nn.Sequential(ResidualBlock(nb_filts=_FILTS[4])),
            nn.Sequential(ResidualBlock(nb_filts=_FILTS[4])),
            nn.Sequential(ResidualBlock(nb_filts=_FILTS[4])),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        self.pos_s = nn.Parameter(torch.randn(1, 42, _FILTS[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, _GAT_DIMS[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, _GAT_DIMS[0]))

        self.gat_layer_s = GraphAttentionLayer(
            _FILTS[-1][-1], _GAT_DIMS[0], temperature=_TEMPERATURES[0]
        )
        self.gat_layer_t = GraphAttentionLayer(
            _FILTS[-1][-1], _GAT_DIMS[0], temperature=_TEMPERATURES[1]
        )
        self.htrg_gat_st11 = HtrgGraphAttentionLayer(
            _GAT_DIMS[0], _GAT_DIMS[1], temperature=_TEMPERATURES[2]
        )
        self.htrg_gat_st12 = HtrgGraphAttentionLayer(
            _GAT_DIMS[1], _GAT_DIMS[1], temperature=_TEMPERATURES[2]
        )
        self.htrg_gat_st21 = HtrgGraphAttentionLayer(
            _GAT_DIMS[0], _GAT_DIMS[1], temperature=_TEMPERATURES[2]
        )
        self.htrg_gat_st22 = HtrgGraphAttentionLayer(
            _GAT_DIMS[1], _GAT_DIMS[1], temperature=_TEMPERATURES[2]
        )

        self.pool_s = GraphPool(_POOL_RATIOS[0], _GAT_DIMS[0], 0.3)
        self.pool_t = GraphPool(_POOL_RATIOS[1], _GAT_DIMS[0], 0.3)
        self.pool_hs1 = GraphPool(_POOL_RATIOS[2], _GAT_DIMS[1], 0.3)
        self.pool_ht1 = GraphPool(_POOL_RATIOS[2], _GAT_DIMS[1], 0.3)
        self.pool_hs2 = GraphPool(_POOL_RATIOS[2], _GAT_DIMS[1], 0.3)
        self.pool_ht2 = GraphPool(_POOL_RATIOS[2], _GAT_DIMS[1], 0.3)

        self.feat_dim = 5 * _GAT_DIMS[1]
        self.out_layer = nn.Linear(self.feat_dim, 2)

    def forward(self, x):
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.ll(x_ssl_feat)  # (bs, frames, 128)

        x = x.transpose(1, 2).unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))

        x = self.encoder(x)
        x = self.selu(self.first_bn1(x))

        w = self.attention(x)

        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_s = m.transpose(1, 2) + self.pos_s
        out_s = self.pool_s(self.gat_layer_s(e_s))

        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_t = m1.transpose(1, 2)
        out_t = self.pool_t(self.gat_layer_t(e_t))

        # Pass the batch-expanded masters to the first HTRG layers too (upstream
        # passes the unexpanded (1, 1, D) parameter and relies on broadcasting).
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_t1, out_s1, master1 = self.htrg_gat_st11(out_t, out_s, master=master1)
        out_s1 = self.pool_hs1(out_s1)
        out_t1 = self.pool_ht1(out_t1)
        out_t_aug, out_s_aug, master_aug = self.htrg_gat_st12(
            out_t1, out_s1, master=master1
        )
        out_t1 = out_t1 + out_t_aug
        out_s1 = out_s1 + out_s_aug
        master1 = master1 + master_aug

        out_t2, out_s2, master2 = self.htrg_gat_st21(out_t, out_s, master=master2)
        out_s2 = self.pool_hs2(out_s2)
        out_t2 = self.pool_ht2(out_t2)
        out_t_aug, out_s_aug, master_aug = self.htrg_gat_st22(
            out_t2, out_s2, master=master2
        )
        out_t2 = out_t2 + out_t_aug
        out_s2 = out_s2 + out_s_aug
        master2 = master2 + master_aug

        out_t1, out_t2 = self.drop_way(out_t1), self.drop_way(out_t2)
        out_s1, out_s2 = self.drop_way(out_s1), self.drop_way(out_s2)
        master1, master2 = self.drop_way(master1), self.drop_way(master2)

        out_t = torch.max(out_t1, out_t2)
        out_s = torch.max(out_s1, out_s2)
        master = torch.max(master1, master2)

        t_max, _ = torch.max(torch.abs(out_t), dim=1)
        t_avg = torch.mean(out_t, dim=1)
        s_max, _ = torch.max(torch.abs(out_s), dim=1)
        s_avg = torch.mean(out_s, dim=1)

        last_hidden = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        logits = self.out_layer(last_hidden)
        return logits
