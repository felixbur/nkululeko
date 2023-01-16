"""
This is a copy of David R. Feinberg's Praat scripts
https://github.com/drfeinberg/PraatScripts
taken June 23rd 2022
"""

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import parselmouth 
import statistics
from nkululeko.util import Util
import audiofile
from parselmouth.praat import call
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#input_wavs = '/home/audeering.local/fburkhardt/audb/emodb/1.1.1/135fc543/wav/'
#input_wavs = './test_wavs/'


# This is the function to measure source acoustics using default male parameters.

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


# ## This function measures formants at each glottal pulse
# 
# Puts, D. A., Apicella, C. L., & Cárdenas, R. A. (2012). Masculine voices signal men's threat potential in forager and industrial societies. Proceedings of the Royal Society of London B: Biological Sciences, 279(1728), 601-609.
# 
# Adapted from: DOI 10.17605/OSF.IO/K2BHS
# This function measures formants using Formant Position formula
#def measureFormants(sound, wave_file, f0min,f0max):
def measureFormants(sound, f0min,f0max):
    sound = parselmouth.Sound(sound) # read the sound
#    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
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
    
    return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median


# ## This function runs a 2-factor Principle Components Analysis (PCA) on Jitter and Shimmer

def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    x = df.loc[:, measures].values
    # f = open('x.pickle', 'wb')
    # pickle.dump(x, f)
    # f.close()

    x = StandardScaler().fit_transform(x)
    if np.any(np.isnan(x)):
        print (f'Warning: {np.count_nonzero(np.isnan(x))} Nans in x, replacing with 0')
        x[np.isnan(x)] = 0
    if np.any(np.isfinite(x)):
        print (f'Warning: {np.count_nonzero(np.isfinite(x))} infinite in x')
    
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print(type(principalComponents))
    if np.any(np.isnan(principalComponents)):
        print ('pc is nan')
        print(f'count: {np.count_nonzero(np.isnan(principalComponents))}')
        print(principalComponents)
        principalComponents=np.nan_to_num(principalComponents)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])

    return principalDf


# ## This block of code runs the above functions on all of the '.wav' files in the /audio folder

def compute_features(file_index):
    # create lists to put the results
    file_list = []
    duration_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
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
    for idx, (wave_file, start, end) in enumerate(file_index.to_list()):
        signal, sampling_rate = audiofile.read(wave_file, offset=start.total_seconds(), duration=(end-start).total_seconds(), always_2d=True)
        sound = parselmouth.Sound(values=signal, sampling_frequency=sampling_rate)
        if idx%10==0:
            print(f'praat: extracting file {idx} of {len(file_index.to_list())}')
        #sound = parselmouth.Sound(wave_file)
        (duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
        localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(
            sound, 75, 300, "Hertz")
        (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median) = measureFormants(
            sound, 75, 300)
#        file_list.append(wave_file) # make an ID list
        duration_list.append(duration) # make duration list
        mean_F0_list.append(meanF0) # make a mean F0 list
        sd_F0_list.append(stdevF0) # make a sd F0 list
        hnr_list.append(hnr) #add HNR data
        
        # add raw jitter and shimmer measures
        localJitter_list.append(localJitter)
        localabsoluteJitter_list.append(localabsoluteJitter)
        rapJitter_list.append(rapJitter)
        ppq5Jitter_list.append(ppq5Jitter)
        ddpJitter_list.append(ddpJitter)
        localShimmer_list.append(localShimmer)
        localdbShimmer_list.append(localdbShimmer)
        apq3Shimmer_list.append(apq3Shimmer)
        aqpq5Shimmer_list.append(aqpq5Shimmer)
        apq11Shimmer_list.append(apq11Shimmer)
        ddaShimmer_list.append(ddaShimmer)
        
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
    df = pd.DataFrame(np.column_stack([duration_list, mean_F0_list, sd_F0_list, hnr_list, 
                                    localJitter_list, localabsoluteJitter_list, rapJitter_list, 
                                    ppq5Jitter_list, ddpJitter_list, localShimmer_list, 
                                    localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
                                    apq11Shimmer_list, ddaShimmer_list, f1_mean_list, 
                                    f2_mean_list, f3_mean_list, f4_mean_list, 
                                    f1_median_list, f2_median_list, f3_median_list, 
                                    f4_median_list]),
                                    columns=['duration', 'meanF0Hz', 'stdevF0Hz', 'HNR', 
                                                'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                                'ppq5Jitter', 'ddpJitter', 'localShimmer', 
                                                'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                                'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean', 
                                                'f3_mean', 'f4_mean', 'f1_median', 
                                                'f2_median', 'f3_median', 'f4_median'])

    # add pca data
    pcaData = runPCA(df) # Run jitter and shimmer PCA
    df = pd.concat([df, pcaData], axis=1) # Add PCA data
    # reload the data so it's all numbers
    df.to_csv("processed_results.csv", index=False)
    df = pd.read_csv('processed_results.csv', header=0)
#    df.sort_values('voiceID').head(20)
    # ## Next we calculate the vocal-tract length estimates

    # ### Formant position
    #  Puts, D. A., Apicella, C. L., & Cárdenas, R. A. (2012). Masculine voices signal men's threat potential in forager and industrial societies. Proceedings of the Royal Society of London B: Biological Sciences, 279(1728), 601-609.

    df['pF'] = (zscore(df.f1_median) + zscore(df.f2_median) + zscore(df.f3_median) + zscore(df.f4_median)) / 4

    # ### Formant Dispersion
    # Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. The Journal of the Acoustical Society of America, 102(2), 1213-1222.


    df['fdisp'] = (df['f4_median'] - df['f1_median']) / 3


    # ### Fn (Average Formant)
    # Pisanski, K., & Rendall, D. (2011). The prioritization of voice fundamental frequency or formants in listeners’ assessments of speaker size, masculinity, and attractiveness. The Journal of the Acoustical Society of America, 129(4), 2201-2212.

    df['avgFormant'] = (df['f1_median'] + df['f2_median'] + df['f3_median'] + df['f4_median']) / 4

    # ### MFF 
    # Smith, D. R., & Patterson, R. D. (2005). The interaction of glottal-pulse rate and vocal-tract length in judgements of speaker size, sex, and age. The Journal of the Acoustical Society of America, 118(5), 3177-3186.

    df['mff'] = (df['f1_median'] * df['f2_median'] * df['f3_median'] * df['f4_median']) ** 0.25


    # ### Fitch VTL
    # Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. The Journal of the Acoustical Society of America, 102(2), 1213-1222.


    # reload the data again
    #df.to_csv("processed_results.csv", index=False)
    #df = pd.read_csv('processed_results.csv', header=0)

    df['fitch_vtl'] = ((1 * (35000 / (4 * df['f1_median']))) +
                    (3 * (35000 / (4 * df['f2_median']))) + 
                    (5 * (35000 / (4 * df['f3_median']))) + 
                    (7 * (35000 / (4 * df['f4_median'])))) / 4


    # ### $\Delta$F 
    # Reby,D.,& McComb,K.(2003). Anatomical constraints generate honesty: acoustic cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.


    xysum = (0.5 * df['f1_median']) + (1.5 * df['f2_median']) + (2.5 * df['f3_median']) + (3.5 * df['f4_median'])
    xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    df['delta_f'] = xysum / xsquaredsum


    # ### VTL($\Delta$F)
    # Reby,D.,&McComb,K.(2003).Anatomical constraints generate honesty: acoustic cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.


    df['vtl_delta_f'] = 35000 / (2 * df['delta_f'])

    return df