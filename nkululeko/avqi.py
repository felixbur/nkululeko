"""Acoustic Voice Quality Index (AVQI v3.01).

Computes the AVQI and its six underlying acoustic measures (CPPS, HNR,
shimmer local, shimmer local dB, LTAS slope, LTAS tilt) from a sustained
vowel (SV) and a continuous speech (CS) recording, following the protocol
of Barsties & Maryn (2015).

Either point this at two existing WAV files (``--sv``/``--cs``), or leave
them out to record both clips interactively via the microphone.

Usage:
    nkululeko.avqi --sv sv.wav --cs cs.wav
    nkululeko.avqi --outdir recordings --outfile avqi_result.csv
"""

# avqi.py
import argparse
import os
import tempfile

import pandas as pd

from nkululeko.constants import VERSION
from nkululeko.utils.util import Util

# Part 0-2 of the original AVQI v3.01 Praat script (Maryn, Corthals,
# Barsties), verbatim, preceded by code that loads the two input files (in
# place of the interactive "already-open Sound sv / Sound cs" assumption of
# the original script). Part 3 (drawing the report/graphs) is omitted since
# only the numeric outputs are needed. This guarantees numerically identical
# results to running the original .praat script in Praat's GUI, without
# needing to touch Praat directly.
_AVQI_SCRIPT = r'''
form AVQI
    sentence sv_path
    sentence cs_path
endform

Read from file: sv_path$
Rename: "sv"
Read from file: cs_path$
Rename: "cs"

# --------------------------------------------------------------------------------------------
# PART 0:
# HIGH-PASS FILTERING OF THE SOUND FILES.
# --------------------------------------------------------------------------------------------

select Sound cs
Filter (stop Hann band)... 0 34 0.1
Rename... cs2
select Sound sv
Filter (stop Hann band)... 0 34 0.1
Rename... sv2

# --------------------------------------------------------------------------------------------
# PART 1:
# DETECTION, EXTRACTION AND CONCATENATION OF
# THE VOICED SEGMENTS IN THE RECORDING
# OF CONTINUOUS SPEECH.
# --------------------------------------------------------------------------------------------

select Sound cs2
Copy... original
samplingRate = Get sampling frequency
intermediateSamples = Get sampling period
Create Sound... onlyVoice 0 0.001 'samplingRate' 0
select Sound original
To TextGrid (silences)... 50 0.003 -25 0.1 0.1 silence sounding
select Sound original
plus TextGrid original
Extract intervals where... 1 no "does not contain" silence
Concatenate
select Sound chain
Rename... onlyLoud
globalPower = Get power in air
select TextGrid original
Remove

select Sound onlyLoud
signalEnd = Get end time
windowBorderLeft = Get start time
windowWidth = 0.03
windowBorderRight = windowBorderLeft + windowWidth
globalPower = Get power in air
voicelessThreshold = globalPower*(30/100)

select Sound onlyLoud
extremeRight = signalEnd - windowWidth
while windowBorderRight < extremeRight
	Extract part... 'windowBorderLeft' 'windowBorderRight' Rectangular 1.0 no
	select Sound onlyLoud_part
	partialPower = Get power in air
	if partialPower > voicelessThreshold
		call checkZeros 0
		if (zeroCrossingRate <> undefined) and (zeroCrossingRate < 3000)
			select Sound onlyVoice
			plus Sound onlyLoud_part
			Concatenate
			Rename... onlyVoiceNew
			select Sound onlyVoice
			Remove
			select Sound onlyVoiceNew
			Rename... onlyVoice
		endif
	endif
	select Sound onlyLoud_part
	Remove
	windowBorderLeft = windowBorderLeft + 0.03
	windowBorderRight = windowBorderLeft + 0.03
	select Sound onlyLoud
endwhile
select Sound onlyVoice

procedure checkZeros zeroCrossingRate

	start = 0.0025
	startZero = Get nearest zero crossing... 'start'
	findStart = startZero
	findStartZeroPlusOne = startZero + intermediateSamples
	startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
	zeroCrossings = 0
	strips = 0

	while (findStart < 0.0275) and (findStart <> undefined)
		while startZeroPlusOne = findStart
			findStartZeroPlusOne = findStartZeroPlusOne + intermediateSamples
			startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
		endwhile
		afstand = startZeroPlusOne - startZero
		strips = strips +1
		zeroCrossings = zeroCrossings +1
		findStart = startZeroPlusOne
	endwhile
	zeroCrossingRate = zeroCrossings/afstand
endproc

# --------------------------------------------------------------------------------------------
# PART 2:
# DETERMINATION OF THE SIX ACOUSTIC MEASURES
# AND CALCULATION OF THE ACOUSTIC VOICE QUALITY INDEX.
# --------------------------------------------------------------------------------------------

select Sound sv2
durationVowel = Get total duration
durationStart=durationVowel-3
if durationVowel>3
Extract part... durationStart durationVowel rectangular 1 no
Rename... sv3
elsif durationVowel<=3
Copy... sv3
endif

select Sound onlyVoice
durationOnlyVoice = Get total duration
plus Sound sv3
Concatenate
Rename... avqi
durationAll = Get total duration
minimumSPL = Get minimum... 0 0 None
maximumSPL = Get maximum... 0 0 None

# Narrow-band spectrogram and LTAS

To Spectrogram... 0.03 4000 0.002 20 Gaussian
select Sound avqi
To Ltas... 1
minimumSpectrum = Get minimum... 0 4000 None
maximumSpectrum = Get maximum... 0 4000 None

# Power-cepstrogram, Cepstral peak prominence and Smoothed cepstral peak prominence

select Sound avqi
To PowerCepstrogram... 60 0.002 5000 50
cpps = Get CPPS... no 0.01 0.001 60 330 0.05 Parabolic 0.001 0 Straight Robust
To PowerCepstrum (slice)... 0.1
maximumCepstrum = Get peak... 60 330 None

# Slope of the long-term average spectrum

select Sound avqi
To Ltas... 1
slope = Get slope... 0 1000 1000 10000 energy

# Tilt of trendline through the long-term average spectrum

select Ltas avqi
Compute trend line... 1 10000
tilt = Get slope... 0 1000 1000 10000 energy

# Amplitude perturbation measures

select Sound avqi
To PointProcess (periodic, cc)... 50 400
Rename... avqi1
select Sound avqi
plus PointProcess avqi1
percentShimmer = Get shimmer (local)... 0 0 0.0001 0.02 1.3 1.6
shim = percentShimmer*100
shdb = Get shimmer (local_dB)... 0 0 0.0001 0.02 1.3 1.6

# Harmonic-to-noise ratio

select Sound avqi
To Pitch (cc)... 0 75 15 no 0.03 0.45 0.01 0.35 0.14 600
select Sound avqi
plus Pitch avqi
To PointProcess (cc)
Rename... avqi2
select Sound avqi
plus Pitch avqi
plus PointProcess avqi2
voiceReport$ = Voice report... 0 0 75 600 1.3 1.6 0.03 0.45
hnr = extractNumber (voiceReport$, "Mean harmonics-to-noise ratio: ")

# Calculation of the AVQI

avqi = (4.152-(0.177*cpps)-(0.006*hnr)-(0.037*shim)+(0.941*shdb)+(0.01*slope)+(0.093*tilt))*2.8902
'''

# Cutoff separating normal from dysphonic voices, as reported for the AVQI
# v3.01 by Barsties & Maryn (2015, https://pubmed.ncbi.nlm.nih.gov/26951063/)
# (sensitivity ~0.92, specificity ~0.90 in the original validation).
# Individual studies report cutoffs roughly in the 2.4-3.2 range depending on
# language/population, so treat this as a rough guide rather than a
# diagnostic threshold.
AVQI_CUTOFF = 2.735

# The AVQI protocol analyzes only the last 3s of the sustained vowel, so
# recordings shorter than that would silently be used in full instead.
MIN_SV_DURATION_S = 3.0
DEFAULT_SV_DURATION_S = 4.0
# A phonetically balanced passage read aloud typically takes 15-25s.
DEFAULT_CS_DURATION_S = 20.0
# The AVQI protocol (Barsties & Maryn, 2015) requires at least 44.1 kHz:
# the LTAS slope/tilt measures analyze the spectrum up to 10 kHz, which
# needs a Nyquist frequency >= 10 kHz. nkululeko's general-purpose default
# of 16 kHz (nkululeko.constants.SAMPLING_RATE) is too low for this and is
# deliberately not used here.
SAMPLE_RATE = 44100

SV_INSTRUCTIONS = (
    "\nSustained vowel recording:\n"
    "  Take a deep breath and sustain the vowel /a:/ at a comfortable pitch\n"
    "  and loudness for the full duration.\n"
)
RAINBOW_PASSAGE_EXCERPT = (
    "When the sunlight strikes raindrops in the air, they act as a prism\n"
    "  and form a rainbow. The rainbow is a division of white light into\n"
    "  many beautiful colors. These take the shape of a long round arch,\n"
    "  with its path high above, and its two ends apparently beyond the\n"
    "  horizon."
)
CS_INSTRUCTIONS = (
    "\nContinuous speech recording:\n"
    "  Read a phonetically balanced passage at a comfortable pitch and\n"
    "  loudness for the full duration, e.g. the opening of the Rainbow\n"
    "  Passage (Fairbanks, 1960):\n\n"
    f"  \"{RAINBOW_PASSAGE_EXCERPT}\"\n"
)


def compute_avqi(sv_path, cs_path):
    """Run the AVQI v3.01 Praat script on the given SV and CS recordings.

    Args:
        sv_path: path to the sustained vowel recording.
        cs_path: path to the continuous speech recording.

    Returns:
        A dict with the six acoustic measures and the resulting avqi score.
    """
    import parselmouth

    sv_path = str(os.path.abspath(sv_path))
    cs_path = str(os.path.abspath(cs_path))

    result = parselmouth.praat.run(
        _AVQI_SCRIPT,
        sv_path,
        cs_path,
        return_variables=True,
    )
    # run() returns (selected_objects, variables_dict) when return_variables=True
    variables = result[-1] if isinstance(result, tuple) else result

    return {
        "cpps": variables["cpps"],
        "hnr": variables["hnr"],
        "shimmer_local": variables["shim"],
        "shimmer_local_db": variables["shdb"],
        "ltas_slope": variables["slope"],
        "ltas_tilt": variables["tilt"],
        "avqi": variables["avqi"],
    }


def _record_clip(seconds, sr, prompt, util, no_playback=False):
    """Record one clip from the microphone, with listen-back and re-record.

    Args:
        seconds: recording duration in seconds.
        sr: sampling rate to record at.
        prompt: text describing what the speaker should do.
        util: Util instance used for warnings.
        no_playback: if True, skip playing the recording back for review.

    Returns:
        The recorded signal as a 1-D numpy array.
    """
    import sounddevice as sd

    print(prompt)
    while True:
        input(f"Press Enter to start recording ({seconds:.0f}s)...")
        print("Recording...", flush=True)
        recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        print("Recording finished.", flush=True)
        if not no_playback:
            try:
                sd.play(recording, sr)
                sd.wait()
            except Exception as e:
                util.warn(f"playback failed: {e}")
        answer = input("Keep this recording? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            return recording.reshape(-1)
        print("Re-recording...")


def _validate_output_path(path, arg_name, util):
    """Validate a user-supplied --outdir/--outfile path before it is used.

    Rejects embedded null bytes (which would truncate the path at the OS
    level) and requires that the resolved path's parent directory already
    exists, so a faulty or crafted argument cannot make this code create or
    write through an arbitrary, unintended directory chain.

    Args:
        path: the raw path supplied via the CLI.
        arg_name: the CLI flag it was passed via (for error messages).
        util: Util instance used to raise on failure.

    Returns:
        The resolved (canonical, absolute) path.
    """
    if "\x00" in path:
        util.error(f"{arg_name} contains an invalid null byte: {path!r}")
    resolved = os.path.realpath(path)
    parent = os.path.dirname(resolved)
    if not os.path.isdir(parent):
        util.error(
            f"{arg_name} parent directory does not exist: {parent} "
            f"(resolved from {path!r})"
        )
    return resolved


def run_interactive(args, util):
    """Obtain the SV and CS recordings (recording interactively as needed).

    Args:
        args: parsed command-line arguments.
        util: Util instance used for logging.

    Returns:
        A dict with the six acoustic measures and the resulting avqi score.
    """
    import audiofile

    needs_recording = args.sv is None or args.cs is None
    outdir = None
    if needs_recording:
        outdir = args.outdir or tempfile.mkdtemp(prefix="nkululeko_avqi_")
        if args.outdir:
            outdir = _validate_output_path(outdir, "--outdir", util)
        os.makedirs(outdir, exist_ok=True)

    sv_path = args.sv
    if sv_path is None:
        signal = _record_clip(
            args.sv_duration, SAMPLE_RATE, SV_INSTRUCTIONS, util, args.no_playback
        )
        sv_path = os.path.join(outdir, "sv.wav")
        audiofile.write(sv_path, signal, SAMPLE_RATE)
        util.debug(f"saved sustained vowel recording to {sv_path}")

    cs_path = args.cs
    if cs_path is None:
        signal = _record_clip(
            args.cs_duration, SAMPLE_RATE, CS_INSTRUCTIONS, util, args.no_playback
        )
        cs_path = os.path.join(outdir, "cs.wav")
        audiofile.write(cs_path, signal, SAMPLE_RATE)
        util.debug(f"saved continuous speech recording to {cs_path}")

    return compute_avqi(sv_path, cs_path)


def interpret_avqi(score):
    """Give a rough, non-diagnostic interpretation of an AVQI score.

    Higher AVQI means worse (more dysphonic) voice quality. Banding is
    centered on the normal/dysphonic cutoff reported for AVQI v3.01 by
    Barsties & Maryn (2015); reported cutoffs vary by language/population
    (roughly 2.4-3.2), so this is only a rough guide.

    Args:
        score: the AVQI value.

    Returns:
        A short human-readable interpretation string.
    """
    if score < AVQI_CUTOFF:
        return "within normal limits (good voice quality)"
    elif score < AVQI_CUTOFF + 1.0:
        return "suggestive of mild dysphonia / reduced voice quality"
    else:
        return "suggestive of moderate-to-severe dysphonia"


def _print_report(results):
    print(f"Smoothed cepstral peak prominence (CPPS): {results['cpps']:.2f}")
    print(f"Harmonics-to-noise ratio (HNR):            {results['hnr']:.2f} dB")
    print(f"Shimmer local:                             {results['shimmer_local']:.2f} %")
    print(f"Shimmer local dB:                          {results['shimmer_local_db']:.2f} dB")
    print(f"Slope of LTAS:                              {results['ltas_slope']:.2f} dB")
    print(f"Tilt of trendline through LTAS:             {results['ltas_tilt']:.2f} dB")
    print(f"AVQI:                                       {results['avqi']:.2f}")
    print(
        f"  -> {interpret_avqi(results['avqi'])} "
        f"(rough guide, normal/dysphonic cutoff ~{AVQI_CUTOFF}; not diagnostic. "
        "See Barsties & Maryn, 2015: https://pubmed.ncbi.nlm.nih.gov/26951063/)"
    )
    print("\nTHIS IS NOT A MEDICAL DEVICE, RESULTS ARE RESEARCH ONLY")


def _check_input_file(path, arg_name, util):
    """Validate a user-supplied --sv/--cs recording before use.

    Checks that the file exists and that its sampling rate meets the AVQI
    protocol's minimum (see SAMPLE_RATE); a lower rate would silently yield
    an incorrect, non-protocol-compliant LTAS slope/tilt instead of an error.

    Args:
        path: path to the recording.
        arg_name: the CLI flag it was passed via (for error messages).
        util: Util instance used to raise on failure.
    """
    import audiofile

    if not os.path.isfile(path):
        util.error(f"{arg_name} file not found: {path}")
    sr = audiofile.sampling_rate(path)
    if sr < SAMPLE_RATE:
        util.error(
            f"{arg_name} file {path} has a sampling rate of {sr} Hz, below "
            f"the {SAMPLE_RATE} Hz required by the AVQI protocol "
            "(Barsties & Maryn, 2015); results would not be valid."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute AVQI v3.01 and its six sub-measures from a sustained "
        "vowel and continuous speech recording (Barsties & Maryn, 2015). "
        "Records both clips interactively via the microphone unless --sv/--cs "
        "point to existing files."
    )
    parser.add_argument(
        "--sv", default=None, help="Path to an existing sustained vowel recording."
    )
    parser.add_argument(
        "--cs", default=None, help="Path to an existing continuous speech recording."
    )
    parser.add_argument(
        "--sv_duration",
        type=float,
        default=DEFAULT_SV_DURATION_S,
        help=f"Seconds to record the sustained vowel for (default: {DEFAULT_SV_DURATION_S}).",
    )
    parser.add_argument(
        "--cs_duration",
        type=float,
        default=DEFAULT_CS_DURATION_S,
        help=f"Seconds to record continuous speech for (default: {DEFAULT_CS_DURATION_S}).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to save interactively recorded audio (default: a temp directory).",
    )
    parser.add_argument(
        "--outfile", default=None, help="Path to save the AVQI results as a CSV file."
    )
    parser.add_argument(
        "--no_playback",
        action="store_true",
        help="Don't play recordings back for review before accepting them.",
    )
    args = parser.parse_args()

    util = Util("avqi", has_config=False)

    if args.sv is not None:
        _check_input_file(args.sv, "--sv", util)
    if args.cs is not None:
        _check_input_file(args.cs, "--cs", util)
    if args.outfile is not None:
        args.outfile = _validate_output_path(args.outfile, "--outfile", util)

    if args.sv is None and args.sv_duration < MIN_SV_DURATION_S:
        util.error(
            f"--sv_duration must be at least {MIN_SV_DURATION_S}s (the AVQI "
            "protocol analyzes only the last 3s of the sustained vowel)"
        )

    util.debug(f"running nkululeko AVQI, version {VERSION}")
    results = run_interactive(args, util)
    _print_report(results)

    if args.outfile:
        pd.DataFrame([results]).to_csv(args.outfile, index=False)
        util.debug(f"saved results to {args.outfile}")

    return results


if __name__ == "__main__":
    main()
