# nkululeko.avqi

The `nkululeko.avqi` module computes the **Acoustic Voice Quality Index
(AVQI v3.01)** and its six underlying acoustic measures (CPPS, HNR, shimmer
local, shimmer local dB, LTAS slope, LTAS tilt) following the protocol of
[Barsties & Maryn (2015)](https://pubmed.ncbi.nlm.nih.gov/26951063/).

It runs the original AVQI v3.01 Praat script (Maryn, Corthals, Barsties)
verbatim, embedded via [parselmouth](https://parselmouth.readthedocs.io/),
which guarantees numerically identical results to running the `.praat`
script in Praat's GUI directly.

> **THIS IS NOT A MEDICAL DEVICE, RESULTS ARE RESEARCH ONLY.** The
> interpretation printed alongside the score is a rough guide, not a
> diagnosis.

## Input recordings

AVQI is computed from two recordings:

1. A **sustained vowel** (SV) — the vowel /a:/ held at a comfortable pitch
   and loudness. Only the last 3 seconds are analyzed, so the recording must
   be at least 3 seconds long.
2. A **continuous speech** (CS) sample — a phonetically balanced passage
   read aloud (e.g. the opening of the *Rainbow Passage*, Fairbanks 1960),
   roughly 15-25 seconds.

You can either point the module at two existing WAV files, or leave `--sv`/
`--cs` out and record both clips interactively via the microphone.

### Sampling rate requirement

AVQI requires recordings sampled at **at least 44.1 kHz**. The LTAS
slope/tilt measures analyze the spectrum up to 10,000 Hz, which needs a
Nyquist frequency of at least 10 kHz; the validated protocol standardizes on
44.1 kHz, 16-bit. This is higher than nkululeko's general-purpose default of
16 kHz, so:

- Interactive recording always records at 44.1 kHz, regardless of the
  general `SAMPLING_RATE` used elsewhere in nkululeko.
- Files passed via `--sv`/`--cs` are checked and rejected if their sampling
  rate is below 44.1 kHz — a lower rate would silently produce an invalid,
  non-protocol-compliant AVQI instead of an error.

## Command-line interface

```text
python -m nkululeko.avqi
    [--sv SV] [--cs CS]
    [--sv_duration SECONDS] [--cs_duration SECONDS]
    [--outdir OUTDIR] [--outfile OUTFILE]
    [--no_playback]
```

| Argument | Description |
|---|---|
| `--sv SV` | Path to an existing sustained vowel recording. If omitted, it is recorded interactively. |
| `--cs CS` | Path to an existing continuous speech recording. If omitted, it is recorded interactively. |
| `--sv_duration SECONDS` | Seconds to record the sustained vowel for (default: `4.0`). Must be at least `3.0`. |
| `--cs_duration SECONDS` | Seconds to record continuous speech for (default: `20.0`). |
| `--outdir OUTDIR` | Directory to save interactively recorded audio (default: a temp directory). Ignored when both `--sv` and `--cs` are given. |
| `--outfile OUTFILE` | Path to save the AVQI results as a CSV file. |
| `--no_playback` | Don't play recordings back for review before accepting them. |

## Examples

### Fully interactive session

```bash
python -m nkululeko.avqi
```

Prompts for the sustained vowel first, then the continuous speech passage,
with a listen-back-and-re-record loop for each. The recorded WAVs are saved
to a temporary directory and the AVQI report is printed to stdout.

### Compute AVQI from existing recordings

```bash
python -m nkululeko.avqi --sv sv.wav --cs cs.wav
```

No recording happens; the two files are validated (existence and sampling
rate) and passed directly to the AVQI computation.

### Record interactively, save recordings and results

```bash
python -m nkululeko.avqi --outdir recordings --outfile avqi_result.csv
```

Saves `recordings/sv.wav`, `recordings/cs.wav` and the six acoustic measures
plus AVQI as a one-row CSV at `avqi_result.csv`.

### Non-interactive recording (no playback)

```bash
python -m nkululeko.avqi --no_playback --sv_duration 5 --cs_duration 25
```

## Output

The printed report includes the six acoustic measures, the AVQI score, and a
rough interpretation banded around the normal/dysphonic cutoff of
`AVQI_CUTOFF = 2.735` reported in the original validation
(sensitivity ~0.92, specificity ~0.90). Reported cutoffs vary by
language/population (roughly 2.4-3.2), so treat the interpretation as
indicative only:

```text
Smoothed cepstral peak prominence (CPPS): 11.11
Harmonics-to-noise ratio (HNR):            16.10 dB
Shimmer local:                             6.01 %
Shimmer local dB:                          0.67 dB
Slope of LTAS:                              -24.39 dB
Tilt of trendline through LTAS:             -10.17 dB
AVQI:                                       3.78
  -> suggestive of moderate-to-severe dysphonia (rough guide, normal/dysphonic
     cutoff ~2.735; not diagnostic. See Barsties & Maryn, 2015:
     https://pubmed.ncbi.nlm.nih.gov/26951063/)

THIS IS NOT A MEDICAL DEVICE, RESULTS ARE RESEARCH ONLY
```

## Python API

```python
from nkululeko.avqi import compute_avqi

results = compute_avqi("sv.wav", "cs.wav")
print(results["avqi"])
```

`compute_avqi()` returns a dict with keys `cpps`, `hnr`, `shimmer_local`,
`shimmer_local_db`, `ltas_slope`, `ltas_tilt`, `avqi`.

## Related

[predict.md](predict.md) (microphone recording for prediction),
[resample.md](resample.md) (sampling rate handling elsewhere in nkululeko).
