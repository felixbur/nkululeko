import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from nkululeko.feat_extract.feats_bert import Bert


@pytest.fixture
def dummy_util(tmp_path):
    # Minimal util mock with required methods
    util = MagicMock()
    util.get_path.return_value = str(tmp_path)
    util.config_val.side_effect = lambda section, key, default=None: default
    util.debug = MagicMock()
    util.error = MagicMock()
    return util


@pytest.fixture
def dummy_data_df():
    # DataFrame with a MultiIndex and a 'text' column
    idx = pd.MultiIndex.from_tuples([("file1",), ("file2",)], names=["file"])
    df = pd.DataFrame({"text": ["Hello world", "Test sentence"]}, index=idx)
    return df


@pytest.fixture
def bert_instance(dummy_data_df, dummy_util):
    # Patch Featureset to inject util and patch the problematic lines in Bert.__init__
    with patch(
        "nkululeko.feat_extract.feats_bert.Featureset.__init__", return_value=None
    ), patch("torch.cuda.is_available", return_value=False):
        bert = Bert.__new__(Bert)  # Create instance without calling __init__
        # Manually set required attributes
        bert.util = dummy_util
        bert.data_df = dummy_data_df
        bert.name = "testbert"
        bert.feat_type = "bert-base-uncased"
        bert.device = "cpu"
        bert.model_initialized = False
        return bert


def test_init_sets_device_and_feat_type(dummy_data_df, dummy_util):
    with patch(
        "nkululeko.feat_extract.feats_bert.Featureset.__init__", return_value=None
    ):
        with patch("torch.cuda.is_available", return_value=False):
            # Create instance and manually set util before accessing it
            bert = Bert.__new__(Bert)
            bert.util = dummy_util
            bert.__init__("testbert", dummy_data_df, "bert")
            # The feat_type now uses the full HuggingFace model path
            assert bert.feat_type in [
                "bert-base-uncased",
                "google-bert/bert-base-uncased",
            ]
            assert bert.model_initialized is False


def test_init_model_calls_transformers(bert_instance):
    with patch("transformers.AutoConfig.from_pretrained") as mock_config, patch(
        "transformers.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch("transformers.AutoModel.from_pretrained") as mock_model:
        mock_cfg = MagicMock()
        mock_cfg.num_hidden_layers = 12
        mock_cfg.tokenizer_class = "BertTokenizer"
        mock_config.return_value = mock_cfg

        # Mock the model properly
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        # Mock tokenizer
        mock_tokenizer.return_value = MagicMock()

        bert_instance.util.config_val.side_effect = (
            lambda section, key, default=None: default
        )
        bert_instance.init_model()
        assert bert_instance.model_initialized is True
        mock_tokenizer.assert_called()
        mock_model.assert_called()


def test_extract_creates_and_loads_pickle(tmp_path, bert_instance):
    # Patch model init and get_embeddings to return dummy vectors
    bert_instance.model_initialized = True
    bert_instance.model_path = "bert-base-uncased"  # Add missing attribute
    bert_instance.get_embeddings = MagicMock(
        side_effect=lambda text, file: np.ones(768)
    )
    bert_instance.util.get_path.return_value = str(tmp_path)

    # Create a more comprehensive config_val mock that returns appropriate values
    def config_val_mock(section, key, default=None):
        if key == "needs_feature_extraction":
            return True  # Force extraction
        elif key == "no_reuse":
            return "False"  # Return string for eval()
        elif key == "bert.text_column":
            return "text"
        else:
            return default

    bert_instance.util.config_val.side_effect = config_val_mock

    # Mock glob_conf to avoid the 'NoneType' error
    with patch("nkululeko.feat_extract.feats_bert.glob_conf") as mock_glob_conf:
        mock_glob_conf.config = {"DATA": {}}

        with patch("os.path.isfile", return_value=False):
            storage = tmp_path / "testbert.pkl"
            # Remove file if exists
            if storage.exists():
                storage.unlink()
            bert_instance.extract()
            # Check that the pickle file was created
            assert storage.exists()
            assert isinstance(bert_instance.df, pd.DataFrame)
            assert bert_instance.df.shape[0] == 2  # 2 rows in dummy_data_df
            assert bert_instance.df.shape[1] == 768  # BERT embedding size

        # Now test loading from cache
        def config_val_mock_cache(section, key, default=None):
            if key == "needs_feature_extraction":
                return False  # Use cache
            elif key == "no_reuse":
                return "False"
            elif key == "bert.model":
                return "bert-base-uncased"
            else:
                return default

        with patch("os.path.isfile", return_value=True):
            with patch("pandas.read_pickle") as mock_read_pickle:
                cached_df = pd.DataFrame({"feat_0": [0.1, 0.2], "feat_1": [0.3, 0.4]})
                mock_read_pickle.return_value = cached_df

                bert_instance2 = bert_instance
                bert_instance2.util.config_val.side_effect = config_val_mock_cache
                bert_instance2.extract()

                # Verify that pd.read_pickle was called
                mock_read_pickle.assert_called_once()
                assert isinstance(bert_instance2.df, pd.DataFrame)


def test_get_embeddings_returns_numpy_array(bert_instance):
    # Create a mock tokenizer output that has a .to() method
    mock_tokenizer_output = MagicMock()
    mock_tokenizer_output.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Patch tokenizer and model
    bert_instance.tokenizer = MagicMock(return_value=mock_tokenizer_output)

    # Create a proper tensor mock that supports mean() operation
    dummy_tensor = torch.ones((1, 5, 768))
    bert_instance.model = MagicMock()
    bert_instance.model.return_value = (dummy_tensor,)  # Return tuple with tensor

    with patch("torch.no_grad"):
        arr = bert_instance.get_embeddings("hello", "file1")
    assert isinstance(arr, np.ndarray)
    assert arr.shape[-1] == 768


def test_extract_sample_calls_init_and_get_embeddings(bert_instance):
    bert_instance.init_model = MagicMock()
    bert_instance.get_embeddings = MagicMock(return_value=np.ones(768))
    feats = bert_instance.extract_sample("sample text")
    bert_instance.init_model.assert_called_once()
    bert_instance.get_embeddings.assert_called_once()
    assert isinstance(feats, np.ndarray)
