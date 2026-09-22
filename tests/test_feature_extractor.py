import pandas as pd

from nkululeko.feature_extractor import FeatureExtractor


class TestFeatureExtractorDispatch:
    """Guards the feats_type -> extractor-class wiring in
    FeatureExtractor._get_feat_extractor_class(), which resolves purely by
    string convention (module name + capitalize() + "Set") -- a typo there
    would silently break at runtime rather than fail a lookup."""

    def _make_extractor(self):
        return FeatureExtractor(pd.DataFrame(), ["voicesauce"], "test_db", "train")

    def test_voicesauce_resolves_to_voicesauceset(self):
        from nkululeko.feat_extract.feats_voicesauce import VoicesauceSet

        extractor = self._make_extractor()
        assert extractor._get_feat_extractor_class("voicesauce") is VoicesauceSet

    def test_praat_still_resolves_to_praatset(self):
        from nkululeko.feat_extract.feats_praat import PraatSet

        extractor = self._make_extractor()
        assert extractor._get_feat_extractor_class("praat") is PraatSet
