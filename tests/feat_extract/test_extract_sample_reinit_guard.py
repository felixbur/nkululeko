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
"""

from unittest.mock import MagicMock, patch

import pytest

from nkululeko.feat_extract.feats_ast import Ast
from nkululeko.feat_extract.feats_bert import Bert
from nkululeko.feat_extract.feats_hubert import Hubert
from nkululeko.feat_extract.feats_mos import MosSet
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
    with patch(
        f"{cls.__module__}.Featureset.__init__", return_value=None
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
