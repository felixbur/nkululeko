"""This is a copy of David R. Feinberg's Praat scripts.
https://github.com/drfeinberg/PraatScripts
taken June 23rd 2022.
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
from sklearn.decomposition import PCA
from tqdm import tqdm

# This is the function to measure source acoustics using default male parameters.


def measure_pitch(voice_id, f0min, f0max, unit):
    sound = parselmouth.Sound(voice_id)  # read the sound
    duration = call(sound, "Get total duration")  # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object
    mean_f0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
    stdev_f0 = call(
        pitch, "Get standard deviation", 0, 0, unit
    )  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsolute_jitter = call(
        point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
    )
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    local_shimmer = call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    localdb_shimmer = call(
        [sound, point_process],
        "Get shimmer (local_dB)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    apq3_shimmer = call(
        [sound, point_process],
        "Get shimmer (apq3)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    aqpq5_shimmer = call(
        [sound, point_process],
        "Get shimmer (apq5)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    apq11_shimmer = call(
        [sound, point_process],
        "Get shimmer (apq11)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    dda_shimmer = call(
        [sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    return (
        duration,
        mean_f0,
        stdev_f0,
        hnr,
        local_jitter,
        localabsolute_jitter,
        rap_jitter,
        ppq5_jitter,
        ddp_jitter,
        local_shimmer,
        localdb_shimmer,
        apq3_shimmer,
        aqpq5_shimmer,
        apq11_shimmer,
        dda_shimmer,
    )


# ## This function measures formants at each glottal pulse
#
# Puts, D. A., Apicella, C. L., & Cárdenas, R. A. (2012). Masculine voices signal men's threat potential in forager and industrial societies. Proceedings of the Royal Society of London B: Biological Sciences, 279(1728), 601-609.
#
# Adapted from: DOI 10.17605/OSF.IO/K2BHS
# This function measures formants using Formant Position formula
# def measureFormants(sound, wave_file, f0min,f0max):
def measure_formants(sound, f0min, f0max):
    sound = parselmouth.Sound(sound)  # read the sound
    #    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    num_points = call(point_process, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    # Measure formants only at glottal pulses
    for point in range(0, num_points):
        point += 1
        t = call(point_process, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
        f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")
        f4 = call(formants, "Get value at time", 4, t, "Hertz", "Linear")
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if str(f1) != "nan"]
    f2_list = [f2 for f2 in f2_list if str(f2) != "nan"]
    f3_list = [f3 for f3 in f3_list if str(f3) != "nan"]
    f4_list = [f4 for f4 in f4_list if str(f4) != "nan"]

    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)

    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)

    return (
        f1_mean,
        f2_mean,
        f3_mean,
        f4_mean,
        f1_median,
        f2_median,
        f3_median,
        f4_median,
    )


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
    # create lists to put the results
    duration_list = []
    mean_f0_list = []
    sd_f0_list = []
    hnr_list = []
    local_jitter_list = []
    localabsolute_jitter_list = []
    rap_jitter_list = []
    ppq5_jitter_list = []
    ddp_jitter_list = []
    local_shimmer_list = []
    localdb_shimmer_list = []
    apq3_shimmer_list = []
    aqpq5_shimmer_list = []
    apq11_shimmer_list = []
    dda_shimmer_list = []
    f1_mean_list = []
    f2_mean_list = []
    f3_mean_list = []
    f4_mean_list = []
    f1_median_list = []
    f2_median_list = []
    f3_median_list = []
    f4_median_list = []
    # Go through all the wave files in the folder and measure all the acoustics
    #    for i, wave_file in enumerate(file_list):
    for idx, (wave_file, start, end) in enumerate(tqdm(file_index.to_list())):
        signal, sampling_rate = audiofile.read(
            wave_file,
            offset=start.total_seconds(),
            duration=(end - start).total_seconds(),
            always_2d=True,
        )
        try:
            sound = parselmouth.Sound(values=signal, sampling_frequency=sampling_rate)
            (
                duration,
                mean_f0,
                stdev_f0,
                hnr,
                local_jitter,
                localabsolute_jitter,
                rap_jitter,
                ppq5_jitter,
                ddp_jitter,
                local_shimmer,
                localdb_shimmer,
                apq3_shimmer,
                aqpq5_shimmer,
                apq11_shimmer,
                dda_shimmer,
            ) = measure_pitch(sound, 75, 300, "Hertz")
            (
                f1_mean,
                f2_mean,
                f3_mean,
                f4_mean,
                f1_median,
                f2_median,
                f3_median,
                f4_median,
            ) = measure_formants(sound, 75, 300)
            #        file_list.append(wave_file) # make an ID list
        except (statistics.StatisticsError, parselmouth.PraatError) as errors:
            print(f"error on file {wave_file}: {errors}")

        duration_list.append(duration)  # make duration list
        mean_f0_list.append(mean_f0)  # make a mean F0 list
        sd_f0_list.append(stdev_f0)  # make a sd F0 list
        hnr_list.append(hnr)  # add HNR data

        # add raw jitter and shimmer measures
        local_jitter_list.append(local_jitter)
        localabsolute_jitter_list.append(localabsolute_jitter)
        rap_jitter_list.append(rap_jitter)
        ppq5_jitter_list.append(ppq5_jitter)
        ddp_jitter_list.append(ddp_jitter)
        local_shimmer_list.append(local_shimmer)
        localdb_shimmer_list.append(localdb_shimmer)
        apq3_shimmer_list.append(apq3_shimmer)
        aqpq5_shimmer_list.append(aqpq5_shimmer)
        apq11_shimmer_list.append(apq11_shimmer)
        dda_shimmer_list.append(dda_shimmer)

        # add the formant data
        f1_mean_list.append(f1_mean)
        f2_mean_list.append(f2_mean)
        f3_mean_list.append(f3_mean)
        f4_mean_list.append(f4_mean)
        f1_median_list.append(f1_median)
        f2_median_list.append(f2_median)
        f3_median_list.append(f3_median)
        f4_median_list.append(f4_median)
    # ## This block of code adds all of that data we just generated to a Pandas data frame
    # Add the data to Pandas
    df = pd.DataFrame(
        np.column_stack(
            [
                duration_list,
                mean_f0_list,
                sd_f0_list,
                hnr_list,
                local_jitter_list,
                localabsolute_jitter_list,
                rap_jitter_list,
                ppq5_jitter_list,
                ddp_jitter_list,
                local_shimmer_list,
                localdb_shimmer_list,
                apq3_shimmer_list,
                aqpq5_shimmer_list,
                apq11_shimmer_list,
                dda_shimmer_list,
                f1_mean_list,
                f2_mean_list,
                f3_mean_list,
                f4_mean_list,
                f1_median_list,
                f2_median_list,
                f3_median_list,
                f4_median_list,
            ]
        ),
        columns=[
            "duration",
            "meanF0Hz",
            "stdevF0Hz",
            "HNR",
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
            "f1_mean",
            "f2_mean",
            "f3_mean",
            "f4_mean",
            "f1_median",
            "f2_median",
            "f3_median",
            "f4_median",
        ],
    )

    # add pca data
    pca_data = run_pca(df)  # Run jitter and shimmer PCA
    df = pd.concat([df, pca_data], axis=1)  # Add PCA data
    # reload the data so it's all numbers
    # df.to_csv("processed_results.csv", index=False)
    # df = pd.read_csv("processed_results.csv", header=0)
    #    df.sort_values('voiceID').head(20)
    # ## Next we calculate the vocal-tract length estimates

    # ### Formant position
    #  Puts, D. A., Apicella, C. L., & Cárdenas, R. A. (2012). Masculine voices signal men's threat potential in forager and industrial societies. Proceedings of the Royal Society of London B: Biological Sciences, 279(1728), 601-609.

    df["pF"] = (
        zscore(df.f1_median)
        + zscore(df.f2_median)
        + zscore(df.f3_median)
        + zscore(df.f4_median)
    ) / 4

    # ### Formant Dispersion
    # Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. The Journal of the Acoustical Society of America, 102(2), 1213-1222.

    df["fdisp"] = (df["f4_median"] - df["f1_median"]) / 3

    # ### Fn (Average Formant)
    # Pisanski, K., & Rendall, D. (2011). The prioritization of voice fundamental frequency or formants in listeners’ assessments of speaker size, masculinity, and attractiveness. The Journal of the Acoustical Society of America, 129(4), 2201-2212.

    df["avgFormant"] = (
        df["f1_median"] + df["f2_median"] + df["f3_median"] + df["f4_median"]
    ) / 4

    # ### MFF
    # Smith, D. R., & Patterson, R. D. (2005). The interaction of glottal-pulse rate and vocal-tract length in judgements of speaker size, sex, and age. The Journal of the Acoustical Society of America, 118(5), 3177-3186.

    df["mff"] = (
        df["f1_median"] * df["f2_median"] * df["f3_median"] * df["f4_median"]
    ) ** 0.25

    # ### Fitch VTL
    # Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. The Journal of the Acoustical Society of America, 102(2), 1213-1222.

    # reload the data again
    # df.to_csv("processed_results.csv", index=False)
    # df = pd.read_csv('processed_results.csv', header=0)

    df["fitch_vtl"] = (
        (1 * (35000 / (4 * df["f1_median"])))
        + (3 * (35000 / (4 * df["f2_median"])))
        + (5 * (35000 / (4 * df["f3_median"])))
        + (7 * (35000 / (4 * df["f4_median"])))
    ) / 4

    # ### $\Delta$F
    # Reby,D.,& McComb,K.(2003). Anatomical constraints generate honesty: acoustic cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.

    xysum = (
        (0.5 * df["f1_median"])
        + (1.5 * df["f2_median"])
        + (2.5 * df["f3_median"])
        + (3.5 * df["f4_median"])
    )
    xsquaredsum = (0.5**2) + (1.5**2) + (2.5**2) + (3.5**2)
    df["delta_f"] = xysum / xsquaredsum

    # ### VTL($\Delta$F)
    # Reby,D.,&McComb,K.(2003).Anatomical constraints generate honesty: acoustic cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.

    df["vtl_delta_f"] = 35000 / (2 * df["delta_f"])

    print("Now extracting speech rate parameters...")

    df_speechrate = get_speech_rate(file_index)
    print("")

    return df.join(df_speechrate)


"""
Speech rate script taken from https://github.com/drfeinberg/PraatScripts
on 25/05/23
"""


def get_speech_rate(file_index):
    cols = [
        "nsyll",
        "npause",
        "dur_s",
        "phonationtime_s",
        "speechrate_nsyll_dur",
        "articulation_rate_nsyll_phonationtime",
        "ASD_speakingtime_nsyll",
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
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

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
    currenttime = timepeaks[0]
    currentint = intensities[0]
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
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    try:
        asd = speakingtot / voicedcount
    except ZeroDivisionError:
        asd = 0
        print("caught zero division")
    speechrate_dictionary = {
        "nsyll": voicedcount,
        "npause": npause,
        "dur_s": originaldur,
        "phonationtime_s": intensity_duration,
        "speechrate_nsyll_dur": speakingrate,
        "articulation_rate_nsyll_phonationtime": articulationrate,
        "ASD_speakingtime_nsyll": asd,
    }
    return speechrate_dictionary
