# augmenter.py
import pandas as pd
from util import Util 
import auglib
from auglib import AudioBuffer
from auglib.utils import random_seed
from auglib.source import Read
from auglib import IntUni, FloatUni, FloatNorm
from auglib.transform import BandPass, BandStop, LowPass, HighPass, \
    ClipByRatio, GainStage, Compose, Select, WhiteNoiseGaussian, PinkNoise
from auglib.core.observe import BoolRand


class Augmenter:
    auglib.utils.random_seed(42)  # Seed the random engine for reproducibility


    def __init__(self, data_df):
        self.data_df = data_df
        self.util = Util()
        # Define a standard transformation that randomly add augmentations to files
        sampling_rate = 16000
        gain = GainStage(
            gain_db=FloatNorm(0, 4, minimum=-12, maximum=12),
        )
        bandpass_center_max = 1600
        bandpass_center_mean = 1500
        bandpass_center_min = 1401
        bandpass_bandwidth_max = 2800
        bandpass_bandwidth_mean = 2400
        bandpass_bandwidth_min = 2000
        eq_filter = Select(
            [
            BandPass(
                center=FloatNorm(
                    mean=bandpass_center_mean, 
                    std=100,
                    minimum=bandpass_center_min,
                    maximum=bandpass_center_max
                ),
                bandwidth=FloatNorm(
                    mean=bandpass_bandwidth_mean,
                    std=100,
                    minimum=bandpass_bandwidth_min,
                    maximum=bandpass_bandwidth_max
                ),
                order=IntUni(low=1, high=2),
                design='butter',
                ),
                LowPass(
                    cutoff=FloatNorm(
                        mean=4500, 
                        std=750,
                        minimum=3000,
                        maximum=6000
                    ),
                    order=IntUni(low=1, high=4),
                    design='butter',
                ),
                HighPass(
                    cutoff=FloatNorm(
                        mean=200, 
                        std=100,
                        minimum=50,
                        maximum=500
                    ),
                    order=IntUni(low=1, high=4),
                    design='butter',
                )
            ],
            bypass_prob=0.7
        )
        bandstop_bandwidth = 5 * sampling_rate / 1024
        bandstop_center_min = 1 + bandstop_bandwidth / 2
        bandstop_center_max = (sampling_rate  / 2) - (bandstop_bandwidth / 2) - 1
        bandstop = BandStop(
            center=FloatUni(bandstop_center_min, bandstop_center_max),
            bandwidth=bandstop_bandwidth,
            order=2,
            bypass_prob=0.66
        )
        noise = Select(
            [
                WhiteNoiseGaussian(gain_db=FloatNorm(-40, 5, maximum=-30)),
                PinkNoise(gain_db=FloatNorm(-30, 5, maximum=-25)),
            ],
            bypass_prob=0.66
        )
        clip = ClipByRatio(
            ratio=FloatNorm(0.05, 0.03, minimum=0.0, maximum=0.08),
            soft=BoolRand(0.5),
            normalize=False,
            bypass_prob=0.8
        )
        self.transforms_chain_speech = Compose([gain, eq_filter, bandstop, noise, clip])

    def augment(self):
        augment = auglib.Augment(
            transform=self.transforms_chain_speech, 
            num_workers=5,  
        )
        df_aug = augment.augment(data=self.data_df, cache_root=self.util.get_path('store'))
        df_aug.index = df_aug.index.droplevel([1,2])
        return df_aug