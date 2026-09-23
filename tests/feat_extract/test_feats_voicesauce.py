import glob
import os
from unittest.mock import MagicMock, patch

import numpy as np

from nkululeko.feat_extract.feats_voicesauce import VoicesauceSet


class TestExtractSampleTempFileSafety:
    """Reviewer follow-up: extract_sample() previously wrote to a fixed
    relative filename, which is unsafe under concurrent calls (one call's
    audio could overwrite another's before compute_features read it) and
    left the file behind in the caller's working directory. It must now use
    a unique temp file that's removed afterwards, even on failure."""

    def _make_extractor(self):
        vs = VoicesauceSet.__new__(VoicesauceSet)
        vs.util = MagicMock()
        vs.util.config_val.return_value = "False"
        vs.util.handle_nan.side_effect = lambda df, context: df
        return vs

    def test_temp_file_is_removed_after_extraction(self, tmp_path):
        vs = self._make_extractor()
        signal = np.zeros((1, 1600), dtype=np.float32)

        before = set(glob.glob(os.path.join(tmp_path, "tmp*")))
        with patch("tempfile.gettempdir", return_value=str(tmp_path)):
            vs.extract_sample(signal, 16000)
        after = set(glob.glob(os.path.join(tmp_path, "tmp*")))

        assert after == before, f"leftover temp files: {after - before}"

    def test_temp_file_is_removed_even_if_extraction_raises(self, tmp_path):
        vs = self._make_extractor()
        signal = np.zeros((1, 1600), dtype=np.float32)

        before = set(glob.glob(os.path.join(tmp_path, "tmp*")))
        with (
            patch("tempfile.gettempdir", return_value=str(tmp_path)),
            patch(
                "nkululeko.feat_extract.feats_voicesauce.feats_voicesauce_core.compute_features",
                side_effect=RuntimeError("boom"),
            ),
        ):
            try:
                vs.extract_sample(signal, 16000)
            except RuntimeError:
                pass
        after = set(glob.glob(os.path.join(tmp_path, "tmp*")))

        assert after == before, f"leftover temp files: {after - before}"

    def test_two_calls_use_different_temp_files(self, tmp_path):
        vs = self._make_extractor()
        signal = np.zeros((1, 1600), dtype=np.float32)

        seen_paths = []
        real_compute_features = (
            __import__(
                "nkululeko.feat_extract.feats_voicesauce_core", fromlist=["compute_features"]
            ).compute_features
        )

        def spy_compute_features(index):
            seen_paths.append(index.to_list()[0][0])
            return real_compute_features(index)

        with (
            patch("tempfile.gettempdir", return_value=str(tmp_path)),
            patch(
                "nkululeko.feat_extract.feats_voicesauce.feats_voicesauce_core.compute_features",
                side_effect=spy_compute_features,
            ),
        ):
            vs.extract_sample(signal, 16000)
            vs.extract_sample(signal, 16000)

        assert len(seen_paths) == 2
        assert seen_paths[0] != seen_paths[1]
