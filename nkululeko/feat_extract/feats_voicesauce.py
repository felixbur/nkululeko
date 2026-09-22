# feats_voicesauce.py
import os
import tempfile

import pandas as pd

from nkululeko.feat_extract import feats_voicesauce_core
from nkululeko.feat_extract.featureset import Featureset


class VoicesauceSet(Featureset):
    """A feature extractor for VoiceSauce-style voice-quality measures.

    Computes H1, H2, H4, A1, A2, A3, their spectral-tilt differences
    (H1H2, H1A1, H1A2, H1A3, H2H4), and CPP -- see feats_voicesauce_core.py
    for what's computed and why this reimplements VoiceSauce's measures via
    parselmouth rather than wrapping the (abandoned, incomplete)
    opensauce-python tool (issue #422).
    """

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        self.print_feats = (
            self.util.config_val("FEATS", "print_feats", "False").lower() == "true"
        )

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"

        if self._needs_extraction(storage):
            self.util.debug(
                "extracting VoiceSauce features, this might take a while..."
            )
            self.df = feats_voicesauce_core.compute_features(self.data_df.index)
            self.df = self.df.set_index(self.data_df.index)
            self.df = self.util.handle_nan(self.df, context="voicesauce features")

            self.util.write_store(self.df, storage, store_format)
            try:
                self.context.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted VoiceSauce features: {storage}.")
            self.df = self.util.get_store(storage, store_format)
        if self.print_feats:
            self.util.debug(f"voicesauce feature names: {self.df.columns}")
        self.df = self.df.astype(float)

    def extract_sample(self, signal, sr):
        import audformat
        import audiofile

        # A fixed relative filename here would be unsafe under concurrent
        # calls (one call's audio can overwrite another's before
        # compute_features reads it) and would litter the caller's working
        # directory. Use a unique file under the system temp directory
        # instead, and always remove it afterwards.
        fd, tmp_audio_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audiofile.write(tmp_audio_name, signal, sr)
            df = pd.DataFrame(index=[tmp_audio_name])
            index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
            df = feats_voicesauce_core.compute_features(index)
            df.set_index(index)
            df = self.util.handle_nan(df, context="voicesauce features")
            df = df.astype(float)
            feats = df.to_numpy()
        finally:
            os.remove(tmp_audio_name)
        return feats
