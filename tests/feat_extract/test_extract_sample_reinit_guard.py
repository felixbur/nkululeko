"""Regression test: extract_sample() must not reload the underlying model
on every call.

`nkululeko.predict --type model` calls `extract_sample()` once per row.
Several Featureset subclasses' `extract_sample()` called `init_model()`
unconditionally (unlike their own `extract()`, which correctly guards with
`if not self.model_initialized`). For a corpus of thousands of rows, that
reloads a full HuggingFace/torch/speechbrain model from scratch on every
single row instead of once -- a severe, silent performance regression with
no error, just "loading ... model" logged (and the download/load repeated)
for every prediction.

`feats_snr.py`'s `SnrSet` had a variant of the same copy-paste: it called
`self.init_model()` even though `SnrSet` (plain SNR estimation, no ML model)
never defines `init_model` or a `model_initialized` flag at all -- so every
call to `extract_sample()` raised `AttributeError`, not just a slow reload.
"""

from unittest.mock import MagicMock

import pytest

from nkululeko.feat_extract.feats_ast import Ast
from nkululeko.feat_extract.feats_bert import Bert
from nkululeko.feat_extract.feats_hubert import Hubert
from nkululeko.feat_extract.feats_mos import MosSet
from nkululeko.feat_extract.feats_snr import SnrSet
from nkululeko.feat_extract.feats_squim import SquimSet
from nkululeko.feat_extract.feats_textclassifier import TextClassifier
from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2
from nkululeko.feat_extract.feats_wavlm import Wavlm
from nkululeko.feat_extract.feats_whisper import Whisper

# (class, flag attribute name, extract_sample args, embedding method name)
CASES = [
    (Wav2vec2, "model_initialized", ("signal", "sr"), "get_embeddings"),
    (Hubert, "model_initialized", ("signal", "sr"), "get_embeddings"),
    (Wavlm, "model_initialized", ("signal", "sr"), "get_embeddings"),
    (Whisper, "model_initialized", ("signal", "sr"), "get_embeddings"),
    (Ast, "model_initialized", ("signal", "sr"), "get_embeddings"),
    (Bert, "model_initialized", ("text",), "get_embeddings"),
    (TextClassifier, "model_initialized", ("text",), "get_results"),
    (MosSet, "model_initialized", ("signal", "sr"), "get_embeddings"),
    (SquimSet, "model_initialized", ("signal", "sr"), "get_embeddings"),
]


try:
    from nkululeko.feat_extract.feats_clap import ClapSet

    CASES.append((ClapSet, "model_initialized", ("signal", "sr"), "get_embeddings"))
except ImportError:
    pass

try:
    from nkululeko.feat_extract.feats_spkrec import Spkrec

    CASES.append(
        (Spkrec, "classifier_initialized", ("signal", "sr"), "get_embeddings")
    )
except ImportError:
    pass


@pytest.mark.parametrize(
    "cls, flag_attr, call_args, embed_method",
    CASES,
    ids=[c[0].__name__ for c in CASES],
)
def test_extract_sample_only_initializes_model_once(
    cls, flag_attr, call_args, embed_method
):
    instance = cls.__new__(cls)
    setattr(instance, flag_attr, False)
    instance.init_model = MagicMock(
        side_effect=lambda: setattr(instance, flag_attr, True)
    )
    setattr(instance, embed_method, MagicMock(return_value="feats"))

    instance.extract_sample(*call_args)
    instance.extract_sample(*call_args)
    instance.extract_sample(*call_args)

    assert instance.init_model.call_count == 1, (
        f"{cls.__name__}.extract_sample() re-initialized the model on a "
        "repeat call instead of reusing the already-loaded one"
    )


def test_snrset_extract_sample_does_not_call_init_model():
    """SnrSet has no model at all -- extract_sample must not reference
    init_model()/model_initialized, which don't exist on this class."""
    instance = SnrSet.__new__(SnrSet)
    instance.get_snr = MagicMock(return_value=12.5)

    result = instance.extract_sample("signal", 16000)

    assert result == 12.5
    instance.get_snr.assert_called_once_with("signal", 16000)
