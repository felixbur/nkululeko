"""Unit tests for the ported AASIST backend (nkululeko/models/model_aasist_core.py).

Builds a tiny randomly-initialized Wav2Vec2Config-based model directly
(no network/download -- same technique as test_model_tuned_backbone.py)
and monkeypatches Wav2Vec2Model.from_pretrained to return it, so
AasistBackend's graph-attention pipeline is exercised end-to-end on real
(if tiny) SSL frontend output shapes without needing the real 300M-param
XLS-R checkpoint.
"""

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from nkululeko.models import model_aasist_core as core


def _tiny_wav2vec2(hidden_size=32):
    config = Wav2Vec2Config(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        attn_implementation="eager",
    )
    return Wav2Vec2Model(config)


class TestHFWav2Vec2Frontend:
    def test_out_dim_matches_checkpoint_hidden_size(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint")

        assert frontend.out_dim == 32

    def test_extract_feat_squeezes_trailing_channel_dim(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint")
        frontend.eval()  # disable dropout so 2D/3D inputs are directly comparable

        x_2d = torch.randn(2, 16000)
        x_3d = x_2d.unsqueeze(-1)

        with torch.no_grad():
            feat_2d = frontend.extract_feat(x_2d)
            feat_3d = frontend.extract_feat(x_3d)

        assert feat_2d.ndim == 3  # (batch, frames, hidden)
        assert feat_2d.shape[0] == 2
        assert feat_2d.shape[-1] == 32
        assert torch.equal(feat_2d, feat_3d)

    def test_weighted_pooling_returns_same_shape_as_last(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", layer_pooling="weighted")
        frontend.eval()

        x = torch.randn(2, 16000)
        with torch.no_grad():
            feat = frontend.extract_feat(x)

        assert feat.shape[0] == 2
        assert feat.shape[-1] == 32

    def test_weighted_pooling_has_one_learnable_weight_per_hidden_state_layer(
        self, monkeypatch
    ):
        # num_hidden_layers=2 -> 2 transformer layers + 1 CNN-extractor
        # output = 3 hidden-state layers total.
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", layer_pooling="weighted")

        assert frontend.layer_weights.shape == (3,)
        assert frontend.layer_weights.requires_grad

    def test_uniform_weights_average_all_layers(self, monkeypatch):
        """With layer_weights left at its zero-init (-> uniform softmax),
        weighted pooling's output must equal the plain mean of every
        hidden-state layer -- confirms the combination math, independent
        of what values training later learns."""
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", layer_pooling="weighted")
        frontend.eval()

        x = torch.randn(2, 16000)
        with torch.no_grad():
            weighted_out = frontend.extract_feat(x)
            hidden_states = frontend.model(x, output_hidden_states=True).hidden_states
            manual_mean = torch.stack(hidden_states, dim=0).mean(dim=0)

        assert torch.allclose(weighted_out, manual_mean, atol=1e-5)

    def test_weighted_pooling_disables_layerdrop(self, monkeypatch):
        """Regression test: wav2vec2's LayerDrop (config.layerdrop, default
        0.1) randomly skips whole encoder layers in train() mode and does
        not append a hidden_states entry when a layer is skipped, so
        len(hidden_states) varies call to call under the default config
        instead of staying fixed at num_hidden_layers + 1 -- breaking
        layer_weights' fixed-size combination with a shape-mismatch
        RuntimeError. Confirmed against the real XLS-R-300M checkpoint
        (22-24 elements instead of 25 across repeated train-mode calls)
        before fixing; this test reproduces the same mechanism on the
        tiny mock model with layerdrop forced high enough to trigger
        reliably, and checks the actual output shape across many calls
        rather than just the config flag, in case a future transformers
        version changes how layerdrop is implemented.
        """
        tiny = _tiny_wav2vec2(hidden_size=32)
        tiny.config.layerdrop = 0.5  # trigger frequently, not just occasionally
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", layer_pooling="weighted")
        assert frontend.model.config.layerdrop == 0.0

        frontend.train()
        x = torch.randn(2, 16000)
        for _ in range(10):
            feat = frontend.extract_feat(x)
            assert feat.shape == (2, feat.shape[1], 32)

    def test_last_pooling_has_no_layer_weights_param(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", layer_pooling="last")

        assert not hasattr(frontend, "layer_weights")

    def test_freeze_disables_gradients_on_frontend_params(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", freeze=True)

        assert all(not p.requires_grad for p in frontend.model.parameters())

    def test_unfrozen_frontend_params_keep_gradients(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", freeze=False)

        assert all(p.requires_grad for p in frontend.model.parameters())

    def test_frozen_frontend_output_has_no_grad_fn(self, monkeypatch):
        """The actual mechanism behind the claimed speedup: a frozen
        frontend's output must carry no autograd history at all (not just
        unused gradients), so no backward graph gets built for it."""
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint", freeze=True)
        frontend.eval()

        x = torch.randn(2, 16000)
        feat = frontend.extract_feat(x)

        assert feat.requires_grad is False


class TestAasistBackendForward:
    def test_forward_returns_two_class_logits(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        backend = core.AasistBackend("dummy/checkpoint")
        backend.eval()

        x = torch.randn(3, 16000)  # 1 second at 16kHz, well under max_len
        with torch.no_grad():
            logits = backend(x)

        assert logits.shape == (3, 2)
        assert torch.all(torch.isfinite(logits))

    def test_forward_handles_trailing_channel_dim_input(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        backend = core.AasistBackend("dummy/checkpoint")
        backend.eval()

        x = torch.randn(2, 16000, 1)
        with torch.no_grad():
            logits = backend(x)

        assert logits.shape == (2, 2)

    def test_forward_with_weighted_layer_pooling_and_frozen_frontend(self, monkeypatch):
        """AasistBackend's layer_pooling/freeze_ssl args must actually
        reach HFWav2Vec2Frontend, not just be accepted and ignored."""
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        backend = core.AasistBackend(
            "dummy/checkpoint", layer_pooling="weighted", freeze_ssl=True
        )
        backend.eval()

        assert backend.ssl_model.layer_pooling == "weighted"
        assert hasattr(backend.ssl_model, "layer_weights")
        assert all(not p.requires_grad for p in backend.ssl_model.model.parameters())

        x = torch.randn(2, 16000)
        with torch.no_grad():
            logits = backend(x)

        assert logits.shape == (2, 2)
        assert torch.all(torch.isfinite(logits))

    def test_feat_dim_matches_out_layer_input_width(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        backend = core.AasistBackend("dummy/checkpoint")

        assert backend.feat_dim == backend.out_layer.in_features


class TestGraphSubmodules:
    def test_graph_attention_layer_preserves_node_count(self):
        layer = core.GraphAttentionLayer(in_dim=8, out_dim=4)
        x = torch.randn(2, 5, 8)
        out = layer(x)
        assert out.shape == (2, 5, 4)

    def test_graph_pool_keeps_ratio_of_nodes(self):
        pool = core.GraphPool(k=0.5, in_dim=4, p=0.0)
        h = torch.randn(2, 10, 4)
        out = pool(h)
        assert out.shape == (2, 5, 4)

    def test_residual_block_changes_channels_via_downsample(self):
        block = core.ResidualBlock(nb_filts=[1, 32], first=True)
        x = torch.randn(2, 1, 20, 20)
        out = block(x)
        assert out.shape[1] == 32

    def test_htrg_graph_attention_layer_shapes(self):
        layer = core.HtrgGraphAttentionLayer(in_dim=8, out_dim=4)
        x1 = torch.randn(2, 3, 8)
        x2 = torch.randn(2, 3, 8)
        out1, out2, master = layer(x1, x2)
        assert out1.shape == (2, 3, 4)
        assert out2.shape == (2, 3, 4)
        assert master.shape == (2, 1, 4)


class TestResidualBlockPreactivation:
    """Deliberate fix to the ported ResidualBlock: non-first blocks apply the
    BN+SELU pre-activation, the first block skips it."""

    def test_non_first_block_applies_preactivation(self):
        torch.manual_seed(0)
        block = core.ResidualBlock([4, 4]).eval()
        x = torch.randn(2, 4, 8, 8) * 5 + 3
        out = block(x)
        expected = block.conv2(
            block.selu(block.bn2(block.conv1(block.selu(block.bn1(x)))))
        )
        assert torch.allclose(out, expected + x, atol=1e-5)

    def test_first_block_skips_preactivation(self):
        torch.manual_seed(0)
        block = core.ResidualBlock([4, 4], first=True).eval()
        x = torch.randn(2, 4, 8, 8)
        expected = block.conv2(block.selu(block.bn2(block.conv1(x))))
        assert torch.allclose(block(x), expected + x, atol=1e-5)
