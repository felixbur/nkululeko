"""
Code originally by Oliver Pauly

Based on an idea by Klaus Scherer

K. R. Scherer, “Randomized splicing: A note on a simple technique for masking speech content” 
Journal of Experimental Research in Personality, vol. 5, pp. 155–159, 1971.

Evaluated in:
F. Burkhardt, Anna Derington, Matthias Kahlau, Klaus Scherer, Florian Eyben and Björn Schuller: Masking Speech Contents by Random Splicing: is Emotional Expression Preserved?, Proc. ICASSP, 2023

"""

import librosa
import numpy as np


def random_splicing(
    signal,
    sr,
    p_reverse=0.0,
    top_db=12,
):
    """
    randomly splice the signal and re-arrange.

    p_reverse: probability of some samples to be in reverse order
    top_db:  top db level for silence to be recognized

    """
    signal /= np.max(abs(signal))

    indices = split_wav_naive(signal, top_db=top_db)

    np.random.shuffle(indices)

    wav_spliced = remix_random_reverse(signal, indices, p_reverse=p_reverse)

    return wav_spliced


def split_wav_naive(wav, top_db=12):
    indices = librosa.effects.split(wav, top_db=top_db)

    indices = np.array(indices)
    # (re)add the silence-segments
    indices = indices.repeat(2)[1:-1].reshape((-1, 2))
    # add first segment
    indices = np.vstack(((0, indices[0][0]), indices))
    # add last segment
    indices = np.vstack((indices, [indices[-1][-1], wav.shape[0]]))

    return indices


def remix_random_reverse(wav, indices, p_reverse=0):

    wav_remix = []

    for seg in indices:
        start = seg[0]
        end = seg[1]
        wav_seg = wav[start:end]

        if np.random.rand(1)[0] <= p_reverse:
            wav_seg = wav_seg[::-1]

        wav_remix.append(wav_seg)

    wav_remix = np.hstack(wav_remix)

    return wav_remix
