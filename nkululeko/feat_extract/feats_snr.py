""" feats_snr.py
    Estimate snr (signal to noise ratio as acoustic features)
"""
import os
from tqdm import tqdm
import pandas as pd
import audiofile
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util
from nkululeko.feat_extract.featureset import Featureset
from nkululeko.autopredict.estimate_snr import SNREstimator


class SNRSet(Featureset):
    """Class to estimate snr"""

    def __init__(self, name, data_df):
        """Constructor."""
        super().__init__(name, data_df)

    def extract(self):
        """Estimate the features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("estimating SNR, this might take a while...")
            snr_series = pd.Series(index=self.data_df.index, dtype=object)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = audiofile.read(
                    file,
                    offset=start.total_seconds(),
                    duration=(end - start).total_seconds(),
                    always_2d=True,
                )
                snr = self.get_snr(signal[0], sampling_rate)
                snr_series[idx] = snr
            print("")
            self.df = pd.DataFrame(snr_series.values.tolist(), index=self.data_df.index)
            self.df.columns = ["snr"]
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing estimated SNR values")
            self.df = self.util.get_store(storage, store_format)

    def get_snr(self, signal, sampling_rate):
        r"""Estimate SNR from raw audio signal.
        Args:
            signal: audio signal
            sampling_rate: sample rate
        Returns
            snr: estimated signal to noise ratio
        """
        snr_estimator = SNREstimator(signal, sampling_rate)
        (
            estimated_snr,
            log_energies,
            energy_threshold_low,
            energy_threshold_high,
        ) = snr_estimator.estimate_snr()
        return estimated_snr

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_snr(signal, sr)
        return feats
