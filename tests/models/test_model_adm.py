"""Unit tests for ADM (Artifact Detection Module) model and core components."""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

from nkululeko.models.model_adm import ADMModel
from nkululeko.models.model_adm_core import (
    TimeADM,
    SpectralADM,
    PhaseADM,
    DeepfakeADMModel,
)


# ============================================================
# Tests for ADM Core Components
# ============================================================


class TestTimeADM:
    """Tests for TimeADM module."""

    def test_init(self):
        """Test TimeADM initialization."""
        adm = TimeADM(feat_dim=768, hidden_dim=256)
        assert adm.fc1.in_features == 768
        assert adm.fc1.out_features == 256
        assert adm.fc2.in_features == 256
        assert adm.fc2.out_features == 256
        assert adm.fc3.in_features == 256
        assert adm.fc3.out_features == 128
        assert adm.fc_out.in_features == 128
        assert adm.fc_out.out_features == 1

    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        adm = TimeADM(feat_dim=768)
        x = torch.randn(4, 768)  # (B, D)
        out = adm(x)
        assert out.shape == (4, 1)

    def test_forward_3d(self):
        """Test forward pass with 3D input (squeezed)."""
        adm = TimeADM(feat_dim=768)
        x = torch.randn(4, 768, 1)  # (B, D, T) with T=1
        out = adm(x)
        assert out.shape == (4, 1)

    def test_dropout_layers(self):
        """Test dropout layers exist."""
        adm = TimeADM(feat_dim=768)
        assert adm.dropout1.p == 0.2
        assert adm.dropout2.p == 0.2
        assert adm.dropout3.p == 0.2

    def test_layer_norm_layers(self):
        """Test layer normalization layers."""
        adm = TimeADM(feat_dim=768, hidden_dim=256)
        assert adm.ln1.normalized_shape == (256,)
        assert adm.ln2.normalized_shape == (256,)
        assert adm.ln3.normalized_shape == (128,)


class TestSpectralADM:
    """Tests for SpectralADM module with LazyLinear."""

    def test_init(self):
        """Test SpectralADM initialization."""
        adm = SpectralADM(hidden_dim=128)
        assert adm.fc2.in_features == 128
        assert adm.fc2.out_features == 128
        assert adm.fc3.in_features == 128
        assert adm.fc3.out_features == 64
        assert adm.fc_out.in_features == 64
        assert adm.fc_out.out_features == 1

    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        adm = SpectralADM()
        x = torch.randn(4, 80)  # (B, F)
        out = adm(x)
        assert out.shape == (4, 1)

    def test_forward_3d(self):
        """Test forward pass with 3D input."""
        adm = SpectralADM()
        x = torch.randn(4, 80, 1)  # (B, F, T)
        out = adm(x)
        assert out.shape == (4, 1)

    def test_lazy_linear_initialization(self):
        """Test that LazyLinear properly initializes on first forward."""
        adm = SpectralADM(hidden_dim=128)
        # Before forward, fc1 is LazyLinear
        assert isinstance(adm.fc1, torch.nn.LazyLinear)

        # After forward, fc1 should be materialized
        x = torch.randn(2, 100)
        _ = adm(x)
        # The lazy module materializes but stays as LazyLinear type
        assert adm.fc1.weight.shape == (128, 100)

    def test_different_input_dims(self):
        """Test SpectralADM with various input dimensions."""
        for feat_dim in [40, 80, 128, 256]:
            adm = SpectralADM()
            x = torch.randn(2, feat_dim)
            out = adm(x)
            assert out.shape == (2, 1)


class TestPhaseADM:
    """Tests for PhaseADM module."""

    def test_init(self):
        """Test PhaseADM initialization."""
        adm = PhaseADM(feat_dim=64, hidden_dim=128)
        # fc1 is LazyLinear, check after forward pass
        assert adm.fc2.in_features == 128
        assert adm.fc2.out_features == 128
        assert adm.fc3.in_features == 128
        assert adm.fc3.out_features == 64
        assert adm.fc_out.in_features == 64
        assert adm.fc_out.out_features == 1

    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        adm = PhaseADM(feat_dim=64)
        x = torch.randn(4, 64)  # (B, D)
        out = adm(x)
        assert out.shape == (4, 1)

    def test_forward_3d(self):
        """Test forward pass with 3D input."""
        adm = PhaseADM(feat_dim=64)
        x = torch.randn(4, 1, 64)  # (B, T, D)
        out = adm(x)
        assert out.shape == (4, 1)


class TestDeepfakeADMModel:
    """Tests for the full DeepfakeADMModel."""

    @pytest.fixture
    def model_weighted(self):
        """Create model with weighted fusion."""
        return DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="weighted",
        )

    @pytest.fixture
    def model_concat(self):
        """Create model with concat fusion."""
        return DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="concat",
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = 4
        ssl_feats = torch.randn(batch_size, 768)
        spec_feats = torch.randn(batch_size, 80)
        phase_feats = torch.randn(batch_size, 64)
        return ssl_feats, spec_feats, phase_feats

    def test_init_weighted(self, model_weighted):
        """Test initialization with weighted fusion."""
        assert model_weighted.fusion == "weighted"
        assert hasattr(model_weighted, "weights")
        assert model_weighted.weights.shape == (3,)

    def test_init_concat(self, model_concat):
        """Test initialization with concat fusion."""
        assert model_concat.fusion == "concat"
        assert hasattr(model_concat, "fusion_fc")
        assert model_concat.fusion_fc.in_features == 3

    def test_forward_weighted(self, model_weighted, sample_inputs):
        """Test forward pass with weighted fusion."""
        ssl_feats, spec_feats, phase_feats = sample_inputs
        out = model_weighted(ssl_feats, spec_feats, phase_feats)
        assert out.shape == (4,)

    def test_forward_concat(self, model_concat, sample_inputs):
        """Test forward pass with concat fusion."""
        ssl_feats, spec_feats, phase_feats = sample_inputs
        out = model_concat(ssl_feats, spec_feats, phase_feats)
        assert out.shape == (4,)

    def test_forward_avg_fusion(self, sample_inputs):
        """Test forward pass with avg fusion."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="avg",
        )
        ssl_feats, spec_feats, phase_feats = sample_inputs
        out = model(ssl_feats, spec_feats, phase_feats)
        assert out.shape == (4,)

    def test_forward_max_fusion(self, sample_inputs):
        """Test forward pass with max fusion."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="max",
        )
        ssl_feats, spec_feats, phase_feats = sample_inputs
        out = model(ssl_feats, spec_feats, phase_feats)
        assert out.shape == (4,)

    def test_invalid_fusion_raises_error(self, sample_inputs):
        """Test that invalid fusion method raises ValueError."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="invalid",
        )
        ssl_feats, spec_feats, phase_feats = sample_inputs
        with pytest.raises(ValueError, match="Unknown fusion method"):
            model(ssl_feats, spec_feats, phase_feats)

    def test_score_method(self, model_weighted, sample_inputs):
        """Test score method (inference mode)."""
        ssl_feats, spec_feats, phase_feats = sample_inputs
        with torch.no_grad():
            out = model_weighted.score(ssl_feats, spec_feats, phase_feats)
        assert out.shape == (4,)

    def test_gradient_flow(self, model_weighted, sample_inputs):
        """Test that gradients flow through the model."""
        ssl_feats, spec_feats, phase_feats = sample_inputs
        ssl_feats.requires_grad_(True)

        out = model_weighted(ssl_feats, spec_feats, phase_feats)
        loss = out.sum()
        loss.backward()

        assert ssl_feats.grad is not None


# ============================================================
# Tests for ADMModel (full model wrapper)
# ============================================================


class DummyUtil:
    """Dummy utility class for testing."""

    def config_val(self, section, key, default=None):
        config_map = {
            ("MODEL", "random_seed"): "False",
            ("MODEL", "loss"): "bce",
            ("MODEL", "device"): "cpu",
            ("MODEL", "learning_rate"): "0.0001",
            ("MODEL", "batch_size"): "2",
            ("MODEL", "optimizer"): "adam",
            ("MODEL", "adm.fusion"): "weighted",
            ("MODEL", "focal.alpha"): "0.25",
            ("MODEL", "focal.gamma"): "2.0",
            ("MODEL", "class_weight"): "auto",
            ("MODEL", "scheduler"): "none",
            ("MODEL", "warmup_epochs"): "5",
            ("MODEL", "scheduler.step_size"): "10",
            ("MODEL", "scheduler.gamma"): "0.5",
            ("MODEL", "max_grad_norm"): "0.0",
            ("MODEL", "feature_noise"): "0.0",
            ("MODEL", "threshold"): "0.5",
            ("EXP", "epochs"): "50",
        }
        return config_map.get((section, key), default)

    def debug(self, msg):
        pass

    def error(self, msg):
        raise Exception(msg)

    def get_path(self, key):
        return "/tmp/"

    def get_exp_name(self, only_train=False):
        return "test_exp"


@pytest.fixture(autouse=True)
def patch_globals(monkeypatch):
    """Patch global config and labels."""
    import nkululeko.glob_conf as glob_conf

    glob_conf.config = {
        "DATA": {"target": "label"},
        "MODEL": {},
    }
    glob_conf.labels = ["real", "fake"]
    yield


@pytest.fixture
def dummy_data():
    """Create dummy data for testing ADM model."""
    # 4 samples, 768 (SSL) + 64 (SPTK) = 832 features
    np.random.seed(42)
    feats_train = pd.DataFrame(
        np.random.rand(4, 832),
        columns=[f"f{i}" for i in range(832)],
    )
    feats_test = pd.DataFrame(
        np.random.rand(2, 832),
        columns=[f"f{i}" for i in range(832)],
    )
    # Use numeric labels (0=real, 1=fake) for binary classification
    df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_test = pd.DataFrame({"label": [1, 0]})
    return df_train, df_test, feats_train, feats_test


@pytest.fixture
def adm_model(dummy_data, monkeypatch):
    """Create ADM model instance for testing."""
    df_train, df_test, feats_train, feats_test = dummy_data

    with patch.object(ADMModel, "__init__", return_value=None):
        model = ADMModel(df_train, df_test, feats_train, feats_test)
        model.util = DummyUtil()
        model.n_jobs = 1
        model.target = "label"
        model.class_num = 2
        model.device = "cpu"
        model.learning_rate = 0.0001
        model.batch_size = 2
        model.num_workers = 1
        model.loss = 0.0
        model.loss_eval = 0.0
        model.run = 0
        model.epoch = 0
        model.df_test = df_test
        model.feats_test = feats_test
        model.feats_train = feats_train
        model.ssl_feat_dim = 768
        model.sptk_feat_dim = 64
        model.store_path = "/tmp/test_adm.pt"

        # Feature indices for splitting (empty means use defaults in _split_features)
        model.fbank_indices = []
        model.stft_indices = []

        # Create BCE loss
        model.criterion = torch.nn.BCEWithLogitsLoss()

        # Create model
        model.model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="weighted",
        ).to("cpu")

        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=0.0001)

        # Set up scheduler (none for testing)
        model.scheduler = None
        model.scheduler_type = "none"
        model.scheduler_needs_init = False

        # Set up training hyperparameters
        model.max_grad_norm = 0.0
        model.feature_noise = 0.0
        model.threshold = 0.5

        # Create data loaders
        model.trainloader = model.get_loader(feats_train, df_train, True)
        model.testloader = model.get_loader(feats_test, df_test, False)

        return model


class TestADMModel:
    """Tests for the ADMModel class."""

    def test_model_init(self, adm_model):
        """Test ADM model initialization."""
        assert hasattr(adm_model, "model")
        assert hasattr(adm_model, "trainloader")
        assert hasattr(adm_model, "testloader")
        assert adm_model.ssl_feat_dim == 768
        assert adm_model.sptk_feat_dim == 64

    def test_split_features(self, adm_model):
        """Test feature splitting."""
        features = torch.randn(4, 832)
        ssl, spec, phase = adm_model._split_features(features)

        assert ssl.shape == (4, 768)
        assert spec.shape == (4, 64)
        assert phase.shape == (4, 64)

    def test_train_one_epoch(self, adm_model):
        """Test training for one epoch."""
        adm_model.train()
        assert adm_model.loss is not None
        assert adm_model.loss >= 0

    def test_evaluate(self, adm_model):
        """Test model evaluation."""
        adm_model.train()
        uar, targets, predictions, logits = adm_model.evaluate(
            adm_model.model, adm_model.testloader, adm_model.device
        )

        assert 0 <= uar <= 1
        assert len(targets) == 2
        assert len(predictions) == 2
        assert len(logits) == 2

    def test_get_predictions(self, adm_model):
        """Test get_predictions method."""
        adm_model.train()
        preds, probas = adm_model.get_predictions()

        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == 2
        assert isinstance(probas, pd.DataFrame)
        assert probas.shape[0] == 2

    def test_get_probas(self, adm_model):
        """Test probability conversion."""
        adm_model.train()
        _, _, _, logits = adm_model.evaluate(
            adm_model.model, adm_model.testloader, adm_model.device
        )
        probas = adm_model.get_probas(logits)

        assert isinstance(probas, pd.DataFrame)
        assert probas.shape[0] == 2
        # Probabilities should sum to ~1 for each sample
        assert np.allclose(probas.sum(axis=1).values, 1.0, atol=1e-5)

    def test_set_testdata(self, adm_model, dummy_data):
        """Test setting new test data."""
        df_train, df_test, feats_train, feats_test = dummy_data
        # Create new test data
        new_df_test = pd.DataFrame({"label": ["real", "fake", "real"]})
        new_feats_test = pd.DataFrame(
            np.random.rand(3, 832),
            columns=[f"f{i}" for i in range(832)],
        )

        adm_model.set_testdata(new_df_test, new_feats_test)

        assert len(adm_model.df_test) == 3
        assert len(adm_model.testloader.dataset) == 3

    def test_reset_test(self, adm_model, dummy_data):
        """Test resetting test data."""
        df_train, df_test, feats_train, feats_test = dummy_data
        adm_model.reset_test(df_test, feats_test)

        assert adm_model.testloader is not None
        assert len(adm_model.testloader.dataset) == 2

    def test_get_loader(self, adm_model, dummy_data):
        """Test data loader creation."""
        df_train, df_test, feats_train, feats_test = dummy_data
        loader = adm_model.get_loader(feats_train, df_train, shuffle=True)

        assert loader is not None
        assert len(loader.dataset) == 4

    def test_store_and_load(self, adm_model, tmp_path):
        """Test model saving and loading."""
        # Set store path to temp directory
        store_path = tmp_path / "test_adm.pt"
        adm_model.store_path = str(store_path)

        # Train and store
        adm_model.train()
        adm_model.store()

        # Verify file exists
        assert store_path.exists()

        # Load into new state dict
        loaded_state = torch.load(str(store_path))
        assert "time_adm.fc1.weight" in loaded_state
        assert "spec_adm.fc2.weight" in loaded_state
        assert "phase_adm.fc1.weight" in loaded_state


class TestADMModelLossFunctions:
    """Tests for different loss function configurations."""

    def test_bce_loss(self, dummy_data, monkeypatch):
        """Test model with BCE loss."""
        from nkululeko.losses.loss_bce import BCEWithLogitsLoss

        df_train, df_test, feats_train, feats_test = dummy_data

        with patch.object(ADMModel, "__init__", return_value=None):
            model = ADMModel(df_train, df_test, feats_train, feats_test)
            model.util = DummyUtil()
            model.criterion = BCEWithLogitsLoss()

            assert isinstance(model.criterion, BCEWithLogitsLoss)

    def test_focal_loss(self, dummy_data, monkeypatch):
        """Test model with Focal loss."""
        from nkululeko.losses.loss_focal import FocalLoss

        df_train, df_test, feats_train, feats_test = dummy_data

        with patch.object(ADMModel, "__init__", return_value=None):
            model = ADMModel(df_train, df_test, feats_train, feats_test)
            model.util = DummyUtil()
            model.criterion = FocalLoss(alpha=0.25, gamma=2.0)

            assert isinstance(model.criterion, FocalLoss)
            assert model.criterion.alpha == 0.25
            assert model.criterion.gamma == 2.0

    def test_weighted_bce_loss(self, dummy_data, monkeypatch):
        """Test model with Weighted BCE loss."""
        from nkululeko.losses.loss_bce import WeightedBCEWithLogitsLoss

        df_train, df_test, feats_train, feats_test = dummy_data

        with patch.object(ADMModel, "__init__", return_value=None):
            model = ADMModel(df_train, df_test, feats_train, feats_test)
            model.util = DummyUtil()
            model.criterion = WeightedBCEWithLogitsLoss(pos_weight=2.0)

            assert isinstance(model.criterion, WeightedBCEWithLogitsLoss)


class TestADMModelFusionMethods:
    """Tests for different fusion methods."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return torch.randn(4, 832)

    def test_weighted_fusion(self, sample_features):
        """Test weighted fusion."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="weighted",
        )
        ssl = sample_features[:, :768]
        sptk = sample_features[:, 768:]

        out = model(ssl, sptk, sptk)
        assert out.shape == (4,)

    def test_concat_fusion(self, sample_features):
        """Test concat fusion."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="concat",
        )
        ssl = sample_features[:, :768]
        sptk = sample_features[:, 768:]

        out = model(ssl, sptk, sptk)
        assert out.shape == (4,)

    def test_avg_fusion(self, sample_features):
        """Test avg fusion."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="avg",
        )
        ssl = sample_features[:, :768]
        sptk = sample_features[:, 768:]

        out = model(ssl, sptk, sptk)
        assert out.shape == (4,)

    def test_max_fusion(self, sample_features):
        """Test max fusion."""
        model = DeepfakeADMModel(
            ssl_feat_dim=768,
            phase_feat_dim=64,
            fusion="max",
        )
        ssl = sample_features[:, :768]
        sptk = sample_features[:, 768:]

        out = model(ssl, sptk, sptk)
        assert out.shape == (4,)
