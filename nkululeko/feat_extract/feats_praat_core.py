"""This is a copy of David R. Feinberg's Praat scripts.
https://github.com/drfeinberg/PraatScripts
taken June 23rd 2022.

2025-05-06: Optimized for faster computation (bta).
"""

#!/usr/bin/env python3
import math
import statistics

import audiofile
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.stats.mstats import zscore
from scipy.stats import lognorm
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm


class AudioFeatureExtractor:
    """Optimized audio feature extraction class to avoid redundant calculations."""

    def __init__(self, f0min=75, f0max=300):
        self.f0min = f0min
        self.f0max = f0max

    def extract_all_features(self, sound):
        """Extract all acoustic features from a single sound object."""
        # Cache common objects to avoid redundant calculations
        duration = sound.get_total_duration()
        pitch = call(sound, "To Pitch", 0.0, self.f0min, self.f0max)
        point_process = call(
            sound, "To PointProcess (periodic, cc)", self.f0min, self.f0max
        )

        # Extract pitch-related features
        pitch_features = self._extract_pitch_features(sound, pitch, point_process)

        # Extract formant features
        formant_features = self._extract_formant_features(sound, point_process)

        # Extract speech rate and pause features
        speech_features = self._extract_speech_features(sound)

        # Combine all features
        all_features = {
            "duration": duration,
            **pitch_features,
            **formant_features,
            **speech_features,
        }

        return all_features

    def _extract_pitch_features(self, sound, pitch, point_process):
        """Extract pitch, jitter, shimmer, and HNR features."""
        # Pitch statistics
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        # HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, self.f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Jitter measures
        local_jitter = call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        localabsolute_jitter = call(
            point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
        )
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer measures (reuse point_process)
        shimmer_params = [0, 0, 0.0001, 0.02, 1.3, 1.6]
        local_shimmer = call(
            [sound, point_process], "Get shimmer (local)", *shimmer_params
        )
        localdb_shimmer = call(
            [sound, point_process], "Get shimmer (local_dB)", *shimmer_params
        )
        apq3_shimmer = call(
            [sound, point_process], "Get shimmer (apq3)", *shimmer_params
        )
        apq5_shimmer = call(
            [sound, point_process], "Get shimmer (apq5)", *shimmer_params
        )
        apq11_shimmer = call(
            [sound, point_process], "Get shimmer (apq11)", *shimmer_params
        )
        dda_shimmer = call([sound, point_process], "Get shimmer (dda)", *shimmer_params)

        return {
            "meanF0Hz": mean_f0,
            "stdevF0Hz": stdev_f0,
            "HNR": hnr,
            "localJitter": local_jitter,
            "localabsoluteJitter": localabsolute_jitter,
            "rapJitter": rap_jitter,
            "ppq5Jitter": ppq5_jitter,
            "ddpJitter": ddp_jitter,
            "localShimmer": local_shimmer,
            "localdbShimmer": localdb_shimmer,
            "apq3Shimmer": apq3_shimmer,
            "apq5Shimmer": apq5_shimmer,
            "apq11Shimmer": apq11_shimmer,
            "ddaShimmer": dda_shimmer,
        }

    def _extract_formant_features(self, sound, point_process):
        """Extract formant features efficiently."""
        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        num_points = call(point_process, "Get number of points")

        # Pre-allocate arrays for better performance
        f1_values = []
        f2_values = []
        f3_values = []
        f4_values = []

        # Single loop to extract all formants
        for point in range(num_points):
            t = call(point_process, "Get time from index", point + 1)
            f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")
            f4 = call(formants, "Get value at time", 4, t, "Hertz", "Linear")

            # Filter out NaN values during collection
            if not math.isnan(f1):
                f1_values.append(f1)
            if not math.isnan(f2):
                f2_values.append(f2)
            if not math.isnan(f3):
                f3_values.append(f3)
            if not math.isnan(f4):
                f4_values.append(f4)

        # Calculate statistics only once
        f1_mean = statistics.mean(f1_values) if f1_values else np.nan
        f2_mean = statistics.mean(f2_values) if f2_values else np.nan
        f3_mean = statistics.mean(f3_values) if f3_values else np.nan
        f4_mean = statistics.mean(f4_values) if f4_values else np.nan

        f1_median = statistics.median(f1_values) if f1_values else np.nan
        f2_median = statistics.median(f2_values) if f2_values else np.nan
        f3_median = statistics.median(f3_values) if f3_values else np.nan
        f4_median = statistics.median(f4_values) if f4_values else np.nan

        return {
            "f1_mean": f1_mean,
            "f2_mean": f2_mean,
            "f3_mean": f3_mean,
            "f4_mean": f4_mean,
            "f1_median": f1_median,
            "f2_median": f2_median,
            "f3_median": f3_median,
            "f4_median": f4_median,
        }

    def _extract_speech_features(self, sound):
        """Extract speech rate and pause features with lognormal distribution analysis."""
        silencedb = -25
        mindip = 2
        minpause = 0.3
        originaldur = sound.get_total_duration()

        # Reuse intensity object for multiple calculations
        intensity = sound.to_intensity(50)
        max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)
        min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

        # Calculate threshold once
        threshold = max_99_intensity + silencedb
        threshold2 = max_intensity - max_99_intensity
        threshold3 = silencedb - threshold2
        if threshold < min_intensity:
            threshold = min_intensity

        # Extract silences and calculate pause durations
        textgrid = call(
            intensity,
            "To TextGrid (silences)",
            threshold3,
            minpause,
            0.1,
            "silent",
            "sounding",
        )
        silencetier = call(textgrid, "Extract tier", 1)
        silencetable = call(silencetier, "Down to TableOfReal", "sounding")
        npauses = call(silencetable, "Get number of rows")

        speakingtot = 0
        pause_durations = []

        # Single loop for speaking time and pause duration calculation
        for ipause in range(npauses):
            pause = ipause + 1
            beginsound = call(silencetable, "Get value", pause, 1)
            endsound = call(silencetable, "Get value", pause, 2)
            speakingdur = endsound - beginsound
            speakingtot += speakingdur

            if ipause > 0:
                prev_endsound = call(silencetable, "Get value", ipause, 2)
                pause_duration = beginsound - prev_endsound
                if pause_duration > 0:
                    pause_durations.append(pause_duration)

        # Calculate pause distribution features
        pause_features = self._calculate_pause_distribution(pause_durations)

        # Efficient syllable counting
        syllable_features = self._count_syllables_optimized(
            sound, intensity, textgrid, threshold, mindip, originaldur
        )

        pausetot = originaldur - speakingtot
        proportion_pause_duration = pausetot / speakingtot if speakingtot > 0 else 0

        return {
            **pause_features,
            **syllable_features,
            "proportion_pause_duration": proportion_pause_duration,
        }

    def _calculate_pause_distribution(self, pause_durations):
        """Calculate lognormal distribution parameters for pause durations."""
        pause_lognorm_mu = np.nan
        pause_lognorm_sigma = np.nan
        pause_lognorm_ks_pvalue = np.nan
        pause_mean_duration = np.nan
        pause_std_duration = np.nan
        pause_cv = np.nan

        if len(pause_durations) >= 3:
            try:
                pause_durations_array = np.array(pause_durations)
                pause_mean_duration = np.mean(pause_durations_array)
                pause_std_duration = np.std(pause_durations_array)
                pause_cv = (
                    pause_std_duration / pause_mean_duration
                    if pause_mean_duration > 0
                    else 0
                )

                shape, loc, scale = lognorm.fit(pause_durations_array, floc=0)
                pause_lognorm_sigma = shape
                pause_lognorm_mu = np.log(scale)

                ks_stat, pause_lognorm_ks_pvalue = stats.kstest(
                    pause_durations_array,
                    lambda x: lognorm.cdf(x, shape, loc=loc, scale=scale),
                )
            except (ValueError, RuntimeError) as e:
                print(f"Error fitting lognormal distribution: {e}")

        return {
            "pause_lognorm_mu": pause_lognorm_mu,
            "pause_lognorm_sigma": pause_lognorm_sigma,
            "pause_lognorm_ks_pvalue": pause_lognorm_ks_pvalue,
            "pause_mean_duration": pause_mean_duration,
            "pause_std_duration": pause_std_duration,
            "pause_cv": pause_cv,
        }

    def _count_syllables_optimized(
        self, sound, intensity, textgrid, threshold, mindip, originaldur
    ):
        """Optimized syllable counting avoiding redundant matrix operations."""
        intensity_matrix = call(intensity, "Down to Matrix")
        sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
        intensity_duration = call(sound_from_intensity_matrix, "Get total duration")

        point_process = call(
            sound_from_intensity_matrix,
            "To PointProcess (extrema)",
            "Left",
            "yes",
            "no",
            "Sinc70",
        )
        numpeaks = call(point_process, "Get number of points")

        # Vectorized time extraction
        timepeaks = []
        intensities = []

        for i in range(numpeaks):
            t = call(point_process, "Get time from index", i + 1)
            value = call(sound_from_intensity_matrix, "Get value at time", t, "Cubic")
            if value > threshold:
                timepeaks.append(t)
                intensities.append(value)

        # Optimized peak validation
        validtime = []
        if len(timepeaks) > 1:
            for p in range(len(timepeaks) - 1):
                currenttime = timepeaks[p]
                currentint = intensities[p]
                dip = call(
                    intensity, "Get minimum", currenttime, timepeaks[p + 1], "None"
                )
                if abs(currentint - dip) > mindip:
                    validtime.append(timepeaks[p])

        # Count voiced syllables
        pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
        voicedcount = 0

        for querytime in validtime:
            whichinterval = call(textgrid, "Get interval at time", 1, querytime)
            whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
            pitch_value = pitch.get_value_at_time(querytime)
            if not math.isnan(pitch_value) and whichlabel == "sounding":
                voicedcount += 1

        # Get silencetable for speaking time calculation
        silencetier = call(textgrid, "Extract tier", 1)
        silencetable = call(silencetier, "Down to TableOfReal", "sounding")
        npauses = call(silencetable, "Get number of rows")

        # Calculate speaking time
        speakingtot = 0
        for i in range(npauses):
            beginsound = call(silencetable, "Get value", i + 1, 1)
            endsound = call(silencetable, "Get value", i + 1, 2)
            speakingtot += endsound - beginsound

        # Calculate rates
        speakingrate = voicedcount / originaldur
        articulationrate = voicedcount / speakingtot if speakingtot > 0 else 0
        asd = speakingtot / voicedcount if voicedcount > 0 else 0

        return {
            "nsyll": voicedcount,
            "npause": npauses - 1,
            "phonationtime_s": intensity_duration,
            "speechrate_nsyll_dur": speakingrate,
            "articulation_rate_nsyll_phonationtime": articulationrate,
            "ASD_speakingtime_nsyll": asd,
        }


# ## This function runs a 2-factor Principle Components Analysis (PCA) on Jitter and Shimmer


def run_pca(df):
    # z-score the Jitter and Shimmer measurements
    measures = [
        "localJitter",
        "localabsoluteJitter",
        "rapJitter",
        "ppq5Jitter",
        "ddpJitter",
        "localShimmer",
        "localdbShimmer",
        "apq3Shimmer",
        "apq5Shimmer",
        "apq11Shimmer",
        "ddaShimmer",
    ]
    x = df.loc[:, measures].values
    # f = open('x.pickle', 'wb')
    # pickle.dump(x, f)
    # f.close()

    # x = StandardScaler().fit_transform(x)
    if np.any(np.isnan(x[0])):
        print(
            f"Warning: {np.count_nonzero(np.isnan(x))} Nans in x, replacing" " with 0"
        )
        x[np.isnan(x)] = 0
    # if np.any(np.isfinite(x[0])):
    #     print(f"Warning: {np.count_nonzero(np.isfinite(x))} finite in x")

    # PCA
    pca = PCA(n_components=2)
    try:
        principal_components = pca.fit_transform(x)
        if np.any(np.isnan(principal_components)):
            print("pc is nan")
            print(f"count: {np.count_nonzero(np.isnan(principal_components))}")
            print(principal_components)
            principal_components = np.nan_to_num(principal_components)
    except ValueError:
        print("need more than one file for pca")
        principal_components = [[0, 0]]
    principal_df = pd.DataFrame(
        data=principal_components, columns=["JitterPCA", "ShimmerPCA"]
    )
    return principal_df


# ## This block of code runs the above functions on all of the '.wav' files in the /audio folder


def compute_features(file_index):
    """Optimized feature computation using AudioFeatureExtractor class.

    FEATURE COUNT COMPARISON:
    Original version: ~36 features
    - Basic: duration, meanF0Hz, stdevF0Hz, HNR (4)
    - Jitter: localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter (5)
    - Shimmer: localShimmer, localdbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer (6)
    - Formants: f1-f4 mean/median (8)
    - PCA: JitterPCA, ShimmerPCA (2)
    - VTL: pF, fdisp, avgFormant, mff, fitch_vtl, delta_f, vtl_delta_f (7)
    - Speech rate: nsyll, npause, phonationtime_s, speechrate_nsyll_dur,
                   articulation_rate_nsyll_phonationtime, ASD_speakingtime_nsyll (6)

    Current optimized version: ~42 features (+6 new pause distribution features)
    - All original 36 features PLUS:
    - Pause distribution: pause_lognorm_mu, pause_lognorm_sigma, pause_lognorm_ks_pvalue,
                         pause_mean_duration, pause_std_duration, pause_cv (6)
    - Additional: proportion_pause_duration (1)

    Total: 43 features (7 new features added for AD detection)
    """
    extractor = AudioFeatureExtractor()
    feature_list = []

    for idx, (wave_file, start, end) in enumerate(tqdm(file_index.to_list())):
        try:
            signal, sampling_rate = audiofile.read(
                wave_file,
                offset=start.total_seconds(),
                duration=(end - start).total_seconds(),
                always_2d=True,
            )
            sound = parselmouth.Sound(values=signal, sampling_frequency=sampling_rate)

            # Extract all features in one pass
            features = extractor.extract_all_features(sound)
            feature_list.append(features)

        except Exception as errors:
            print(f"error on file {wave_file}: {errors}")
            # Add empty feature dict for failed files
            feature_list.append(
                {
                    key: np.nan
                    for key in ["duration", "meanF0Hz", "stdevF0Hz", "HNR"]
                    + [
                        f"f{i}_{stat}"
                        for i in range(1, 5)
                        for stat in ["mean", "median"]
                    ]
                    + [
                        "localJitter",
                        "localabsoluteJitter",
                        "rapJitter",
                        "ppq5Jitter",
                        "ddpJitter",
                        "localShimmer",
                        "localdbShimmer",
                        "apq3Shimmer",
                        "apq5Shimmer",
                        "apq11Shimmer",
                        "ddaShimmer",
                    ]
                }
            )

    # Create DataFrame directly from feature list
    df = pd.DataFrame(feature_list)

    # Add derived features efficiently
    df = add_derived_features(df)

    print(
        f"Feature extraction completed. Total features extracted: {len(df.columns) if 'df' in locals() else '~43'}"
    )
    return df


def add_derived_features(df):
    """Add PCA and vocal tract length features efficiently."""
    # PCA on jitter/shimmer
    pca_data = run_pca(df)
    df = pd.concat([df, pca_data], axis=1)

    # Vectorized vocal tract calculations
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pF"] = (
            zscore(df.f1_median)
            + zscore(df.f2_median)
            + zscore(df.f3_median)
            + zscore(df.f4_median)
        ) / 4
        df["fdisp"] = (df["f4_median"] - df["f1_median"]) / 3
        df["avgFormant"] = (
            df["f1_median"] + df["f2_median"] + df["f3_median"] + df["f4_median"]
        ) / 4
        df["mff"] = (
            df["f1_median"] * df["f2_median"] * df["f3_median"] * df["f4_median"]
        ) ** 0.25

        # Fitch VTL calculation
        df["fitch_vtl"] = (
            (1 * (35000 / (4 * df["f1_median"])))
            + (3 * (35000 / (4 * df["f2_median"])))
            + (5 * (35000 / (4 * df["f3_median"])))
            + (7 * (35000 / (4 * df["f4_median"])))
        ) / 4

        # Delta F calculation
        xysum = (
            0.5 * df["f1_median"]
            + 1.5 * df["f2_median"]
            + 2.5 * df["f3_median"]
            + 3.5 * df["f4_median"]
        )
        xsquaredsum = 0.5**2 + 1.5**2 + 2.5**2 + 3.5**2
        df["delta_f"] = xysum / xsquaredsum
        df["vtl_delta_f"] = 35000 / (2 * df["delta_f"])

    return df


"""
Speech rate script taken from https://github.com/drfeinberg/PraatScripts
on 25/05/23
"""


def get_speech_rate(file_index):
    cols = [
        "nsyll",
        "npause",
        "phonationtime_s",
        "speechrate_nsyll_dur",
        "articulation_rate_nsyll_phonationtime",
        "ASD_speakingtime_nsyll",
        "pause_lognorm_mu",
        "pause_lognorm_sigma",
        "pause_lognorm_ks_pvalue",
        "pause_mean_duration",
        "pause_std_duration",
        "pause_cv",
    ]
    datalist = []
    for idx, (wave_file, start, end) in enumerate(tqdm(file_index.to_list())):
        signal, sampling_rate = audiofile.read(
            wave_file,
            offset=start.total_seconds(),
            duration=(end - start).total_seconds(),
            always_2d=True,
        )
        try:
            sound = parselmouth.Sound(values=signal, sampling_frequency=sampling_rate)
            # print(f'processing {file}')
            speechrate_dictionary = speech_rate(sound)
            datalist.append(speechrate_dictionary)
        except IndexError as ie:
            print(f"error extracting speech-rate on file {wave_file}: {ie}")
        except parselmouth.PraatError as pe:
            print(f"error extracting speech-rate on file {wave_file}: {pe}")
    df = pd.DataFrame(datalist)
    return df


def speech_rate(sound):
    silencedb = -25
    mindip = 2
    minpause = 0.3
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(
        intensity,
        "To TextGrid (silences)",
        threshold3,
        minpause,
        0.1,
        "silent",
        "sounding",
    )
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    pause_durations = []  # Store individual pause durations

    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

        # Calculate pause duration (time between speaking segments)
        if ipause > 0:
            prev_pause = ipause
            prev_endsound = call(silencetable, "Get value", prev_pause, 2)
            pause_duration = beginsound - prev_endsound
            if pause_duration > 0:  # Only include positive pause durations
                pause_durations.append(pause_duration)

    # Calculate pause duration distribution parameters
    pause_lognorm_mu = np.nan
    pause_lognorm_sigma = np.nan
    pause_lognorm_ks_pvalue = np.nan
    pause_mean_duration = np.nan
    pause_std_duration = np.nan
    pause_cv = np.nan

    if len(pause_durations) >= 3:  # Need minimum samples for distribution fitting
        try:
            # Fit lognormal distribution to pause durations
            pause_durations_array = np.array(pause_durations)

            # Calculate basic statistics
            pause_mean_duration = np.mean(pause_durations_array)
            pause_std_duration = np.std(pause_durations_array)
            pause_cv = (
                pause_std_duration / pause_mean_duration
                if pause_mean_duration > 0
                else 0
            )

            # Fit lognormal distribution
            shape, loc, scale = lognorm.fit(pause_durations_array, floc=0)
            pause_lognorm_sigma = shape  # shape parameter (sigma)
            pause_lognorm_mu = np.log(scale)  # location parameter (mu)

            # Test goodness of fit using Kolmogorov-Smirnov test
            ks_stat, pause_lognorm_ks_pvalue = stats.kstest(
                pause_durations_array,
                lambda x: lognorm.cdf(x, shape, loc=loc, scale=scale),
            )

        except (ValueError, RuntimeError) as e:
            print(f"Error fitting lognormal distribution to pause durations: {e}")

    # Calculate pause duration
    pausetot = originaldur - speakingtot

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(
        sound_from_intensity_matrix,
        "To PointProcess (extrema)",
        "Left",
        "yes",
        "no",
        "Sinc70",
    )
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0] if timepeaks else 0
    currentint = intensities[0] if intensities else 0
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime)
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = voicedpeak[i] * timecorrection
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot if speakingtot > 0 else 0
    npause = npauses - 1
    try:
        asd = speakingtot / voicedcount
    except ZeroDivisionError:
        asd = 0
        print("caught zero division")

    # Calculate proportion pause duration
    try:
        proportion_pause_duration = pausetot / speakingtot
    except ZeroDivisionError:
        proportion_pause_duration = 0
        print("caught zero division for proportion pause duration")

    speechrate_dictionary = {
        "nsyll": voicedcount,
        "npause": npause,
        "phonationtime_s": intensity_duration,
        "speechrate_nsyll_dur": speakingrate,
        "articulation_rate_nsyll_phonationtime": articulationrate,
        "ASD_speakingtime_nsyll": asd,
        "proportion_pause_duration": proportion_pause_duration,
        "pause_lognorm_mu": pause_lognorm_mu,
        "pause_lognorm_sigma": pause_lognorm_sigma,
        "pause_lognorm_ks_pvalue": pause_lognorm_ks_pvalue,
        "pause_mean_duration": pause_mean_duration,
        "pause_std_duration": pause_std_duration,
        "pause_cv": pause_cv,
    }
    return speechrate_dictionary
