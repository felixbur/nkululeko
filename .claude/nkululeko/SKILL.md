# Project Skill: nkululeko

## Overview
Nkululeko is a software to detect speaker characteristics by machine learning experiments with a high-level interface. The idea is to have a framework (based on e.g. sklearn and torch) that can be used to rapidly and automatically analyse audio data and explore machine learning models based on that data.

Some abilities that Nkululeko provides: combines acoustic features and machine learning models (including feature selection and features concatenation); performs data exploration, selection and visualization the results; finetuning; ensemble learning models; soft labeling (predicting labels with pre-trained model); and inference the model on a test set.

Nkululeko orchestrates data loading, feature extraction, and model training, allowing you to specify your experiment in a configuration file. The framework handles the process from raw data to trained model and evaluation, making it easy to run machine learning experiments without directly coding in Python.

## Tech Stack
- Python > 3.10

## Code Conventions
- Add tests for new functions
- Add utility functions to the utils sub-package
- Add docstrings to new functions
- Follow PEP8 style guide

## Project Structure
Nkululeko is organized into several modules:
- nkululeko: main module containing core functionality for data loading, feature extraction, model training, and evaluation.
- explore: module for data exploration and visualization.
- predict: module for making predictions with trained models and existing feature extractors.
- augment: module for data augmentation techniques.
- segment: module for audio segmentation and related utilities.
- flags: module for running experiments with different combinations of parameters specified in configuration files.

All models are in the nkululeko.models subpackage, and all feature extractors are in the nkululeko.features subpackage.

## Result Folder Structure
The results of experiments are stored in a structured way under the root directory specified in the configuration file. The structure is as follows:
- root/
  - experiment_name/
    - images/ # for plots and visualizations
    - models/ # for trained model files
    - logs/ # for training logs and metrics
    - results/ # for evaluation results and metrics
    - store/ # for reusable data like extracted features and predictions
    - cache/ # for temporary files and intermediate results

Usually the result files follow a naming convention that includes the experiment name, dataset names, model type, and relevant parameters for easy identification.

## Documentation
Documentation is under the folder docs/source.

All key-value pairs in the configuration files should be documented in the ini_file.md file, which serves as a reference for users to understand the available options and their effects on the experiments.

## Result Analysis

The results of an experiment are in the results folder. When asked to write an exploration report, compile the statistical results including images in a markdown file, but only if the results are statistically significant. If not, just write a short note about the results and do not include any images.

**Always read each distribution image before writing its interpretation.** Do not infer the direction of an effect (e.g. "group A is higher than group B") from the feature name alone — read the actual bar/violin plot to get the correct direction. Getting the direction wrong undermines the whole report.

**Demographic/covariate distributions come first.** Before listing acoustic feature distributions, add the following subsections at the top of each target section:

1. **Age** — check for `value_counts_{target}_age.txt`; if significant, include images (`{target}-age_samples.png`, `{target}-age_speakers.png`) and pairwise pairs. Do not omit even if the main task is about acoustic features.
2. **Gender** — always include the samples plot (`{target}-gender_samples.png`) regardless of significance. No significance text is needed if no stats were computed.

### Praat feature interpretation (acoustic biomarker guide)

Source: `nkululeko/feat_extract/feats_praat_core.py` (David R. Feinberg's PraatScripts, adapted).

**Fundamental frequency**

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `meanF0Hz` | Mean F0 in Hz | Average pitch level. Elevated in high-arousal states (anger, fear); lowered in sadness, depression, Parkinson's. |
| `stdevF0Hz` | Std dev of F0 in Hz | Pitch variability. Low = monotone voice (depression, Parkinson's, flat affect); high = emotional or dysphonic variation. |

**Harmonics-to-Noise Ratio**

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `HNR` | Mean HNR in dB | Ratio of periodic to aperiodic energy. Lower values = more noise (breathiness, roughness, dysphonia). |

**Jitter (period perturbation) — all derived from the glottal pulse PointProcess**

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `localJitter` | Cycle-to-cycle period variation (%) | Primary jitter marker; elevated in roughness and laryngeal pathology. |
| `localabsoluteJitter` | Local jitter in seconds (absolute) | Period perturbation in absolute time; sensitive to very slow voices. |
| `rapJitter` | Relative average perturbation (3-point) | Jitter averaged over 3 consecutive cycles; smoother than local. |
| `ppq5Jitter` | 5-point period perturbation quotient | Smoothed over 5 cycles; preferred in clinical voice research. |
| `ddpJitter` | Difference of differences of periods (= 3 × rap) | Rate of change of jitter; sensitive to rapid period irregularity. |
| `JitterPCA` | PC1 of all 5 jitter measures | Composite jitter index; reduces redundancy. |

**Shimmer (amplitude perturbation)**

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `localShimmer` | Cycle-to-cycle amplitude variation (%) | Primary shimmer marker; elevated in breathiness, weakness, laryngeal pathology. |
| `localdbShimmer` | Local shimmer in dB | Log-scale amplitude perturbation; less sensitive to very large outliers. |
| `apq3Shimmer` | Amplitude perturbation quotient (3-point) | Smoothed shimmer over 3 cycles. |
| `apq5Shimmer` | Amplitude perturbation quotient (5-point) | Smoothed shimmer over 5 cycles. |
| `apq11Shimmer` | Amplitude perturbation quotient (11-point) | Widely used clinical shimmer measure. |
| `ddaShimmer` | Difference of differences of amplitudes (= 3 × apq3) | Rate of shimmer change. |
| `ShimmerPCA` | PC1 of all 6 shimmer measures | Composite shimmer index. |

**Formants** (measured at glottal pulse times using Burg algorithm)

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `f1_mean/median` | F1 mean / median in Hz | First resonance: vowel height and backness; also relates to vocal tract length (VTL). |
| `f2_mean/median` | F2 mean / median in Hz | Second resonance: vowel frontness; speaker identity and accent. |
| `f3_mean/median` | F3 mean / median in Hz | Third resonance: individualised, correlated with VTL. |
| `f4_mean/median` | F4 mean / median in Hz | Fourth resonance: used mainly in VTL estimation. |

**Vocal tract length (VTL) estimates** (all derived from F1–F4 median)

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `pF` | Mean z-scored sum of F1–F4 | Formant position; correlated with body size and age. |
| `fdisp` | (F4 − F1) / 3 | Formant dispersion; inversely related to VTL. |
| `avgFormant` | (F1+F2+F3+F4) / 4 | Average resonance level. |
| `mff` | Geometric mean of F1–F4 | VTL estimator based on formant product. |
| `fitch_vtl` | Weighted sum VTL estimate (Fitch) | Absolute VTL in cm/10 (using 35000 cm/s speed of sound). |
| `delta_f` | Linear regression spacing of F1–F4 | Uniform formant interval estimate. |
| `vtl_delta_f` | 35000 / (2 × delta_f) | VTL from uniform spacing model. |

**Speech rate and rhythm**

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `nsyll` | Number of voiced syllables | Utterance length / speaking activity. |
| `npause` | Number of pauses (> 300 ms) | Hesitation frequency; elevated in cognitive load, Alzheimer's, anxiety. |
| `phonationtime_s` | Total phonation time in seconds | Time spent vocalising. |
| `speechrate_nsyll_dur` | nsyll / total duration | Overall speech rate; slows in depression, Parkinson's, cognitive decline. |
| `articulation_rate_nsyll_phonationtime` | nsyll / phonation time | Motor articulation speed (excluding pauses). |
| `ASD_speakingtime_nsyll` | phonation time / nsyll | Average syllable duration; longer = slower articulation. |
| `proportion_pause_duration` | Total pause time / speaking time | Pause burden; elevated in fluency disorders and cognitive impairment. |

**Pause distribution** (lognormal fit; NaN when fewer than 3 pauses are detected)

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `pause_mean_duration` | Mean pause duration (s) | Average gap length; longer pauses = more retrieval difficulty or motor slowing. |
| `pause_std_duration` | Std dev of pause durations | Variability of gap lengths. |
| `pause_cv` | std / mean of pause durations | Coefficient of variation; irregular pausing (high CV) vs. regular hesitation (low CV). |
| `pause_lognorm_mu` | μ of lognormal fit to pause durations | Location parameter of the pause duration distribution. |
| `pause_lognorm_sigma` | σ (shape) of lognormal fit | Spread; high sigma = long-tail distribution with occasional very long pauses. |
| `pause_lognorm_ks_pvalue` | KS goodness-of-fit p-value | How well pause durations follow a lognormal distribution (low p = poor fit). |

### OpenSMILE eGeMAPS feature interpretation (acoustic biomarker guide)

Source: eGeMAPS v02 (Eyben et al. 2015, IEEE TASLP).  
Paper: https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf  
Config: https://github.com/audeering/opensmile-python/blob/main/opensmile/core/config/egemaps/v02/eGeMAPSv02_core.func.conf.inc

**Naming convention**

| Suffix | Meaning |
|--------|---------|
| `_sma3` | 3-frame moving average; smoothing can cross zero values (`noZeroSma=0`) — used for energy/spectral LLDs |
| `_sma3nz` | 3-frame moving average that does NOT cross zero values (`noZeroSma=1`) — smoothing stops at voiced/unvoiced boundaries; used for F0 and voice-quality LLDs |
| `_amean` | Arithmetic mean over utterance (functional) |
| `_stddevNorm` | Standard deviation normalised by mean (coefficient of variation) — variability / dynamic range |
| `_percentile20/50/80` | 20th / 50th / 80th percentile within utterance |
| `_pctlrange0-2` | p80 − p20 — within-utterance spread |
| `_meanRisingSlope` | Mean slope of rising segments (positive rate of change) |
| `_stddevRisingSlope` | Variability of rising slopes |
| `_meanFallingSlope` | Mean slope of falling segments (negative rate of change) |
| `_stddevFallingSlope` | Variability of falling slopes |
| `V` (e.g. `alphaRatioV`) | Voiced-frame variant |
| `UV` (e.g. `alphaRatioUV`) | Unvoiced-frame variant |

**Fundamental frequency** (`F0semitoneFrom27.5Hz_sma3nz_*` — voiced frames, semitones re 27.5 Hz)

| Feature | Biomarker interpretation |
|---------|--------------------------|
| `_amean` | Mean pitch. Elevated in high-arousal emotions (anger, fear); lowered in sadness, depression, Parkinson's. |
| `_stddevNorm` | Normalized pitch variability. Low = monotone (depression, flat affect); high = emotional expressivity. |
| `_percentile20/50/80` | Lower / median / upper pitch within utterance — characterise pitch distribution shape. |
| `_pctlrange0-2` | Within-utterance pitch range. Narrow = monotone; wide = expressive or dysphonically unstable. |
| `_meanRisingSlope` | Mean steepness of F0 rises. Anger shows abrupt rises; questions and exclamations show high rising slopes. |
| `_stddevRisingSlope` | Variability of F0 rise speed — irregular intonation dynamics. |
| `_meanFallingSlope` | Mean steepness of F0 falls. Declarative statements end with steep falls; depression may show reduced falling slopes. |
| `_stddevFallingSlope` | Variability of F0 fall speed. |

**Loudness** (`loudness_sma3_*` — all frames, PLP-based auditory spectrum with cube-root amplitude compression)

| Feature | Biomarker interpretation |
|---------|--------------------------|
| `_amean` | Mean vocal loudness / effort. Elevated in anger and high-arousal states; reduced in depression, asthenia. |
| `_stddevNorm` | Loudness dynamics. Low = flat, monotone delivery; high = expressive or erratic. |
| `_percentile20/50/80` | Loudness distribution shape within utterance. |
| `_pctlrange0-2` | Loudness range. Narrow in depression/apathy; wide in expressive speech. |
| `_meanRisingSlope` | Rate of loudness increases — emphasis, emotional crescendos. |
| `_stddevRisingSlope` | Variability of loudness ramp-ups. |
| `_meanFallingSlope` | Rate of loudness decreases — breath support run-out, phrase endings. |
| `_stddevFallingSlope` | Variability of loudness decays. |

**Spectral flux** (`spectralFlux_sma3_*` — all frames; `spectralFluxV/UV_sma3nz_*` — voiced/unvoiced)

| Feature | Biomarker interpretation |
|---------|--------------------------|
| `spectralFlux_sma3_amean` | Mean frame-to-frame spectral change. Higher = more dynamic articulation; lower = stable/monotone. |
| `spectralFlux_sma3_stddevNorm` | Variability of spectral flux — intermittent vs. continuous articulation change. |
| `spectralFluxV_sma3nz_amean/stddevNorm` | Same restricted to voiced frames — reflects voiced consonant and vowel transitions. |
| `spectralFluxUV_sma3nz_amean` | Flux in unvoiced frames — captures frication noise dynamics. |

**MFCCs** (`mfcc1-4_sma3_*` all frames; `mfcc1-4V_sma3nz_*` voiced frames only)

| Feature | Biomarker interpretation |
|---------|--------------------------|
| `mfcc1_*` | Captures overall spectral tilt / log energy shape — correlated with vocal tract length and effort. |
| `mfcc2_*` | Second cepstral coefficient — primary vowel formant structure, front-back tongue position. |
| `mfcc3_*` | Third cepstral coefficient — further spectral detail; contributes to vowel and speaker identity. |
| `mfcc4_*` | Fourth cepstral coefficient — fine spectral shape. |
| `_amean` | Typical spectral shape across utterance. |
| `_stddevNorm` | Dynamic variability — how much spectral shape changes, i.e. articulatory range. |
| `V` variants | Same features measured only on voiced frames — less influenced by background noise and unvoiced consonants. |

**Voice quality — jitter, shimmer, HNR** (voiced frames)

| Feature | Biomarker interpretation |
|---------|--------------------------|
| `jitterLocal_sma3nz_amean` | Average absolute local jitter per pitch period within 60 ms frames, normalised by mean period length. Elevated in roughness, laryngeal pathology. Pitch-period-based within frames (differs from Praat's whole-utterance calculation). |
| `jitterLocal_sma3nz_stddevNorm` | Temporal variability of jitter — intermittent vs. sustained irregularity. |
| `shimmerLocaldB_sma3nz_amean` | Relative peak-amplitude differences in dB, averaged over 60 ms frames synchronised to pitch periods. Elevated in breathiness, weakness, laryngeal disease. |
| `shimmerLocaldB_sma3nz_stddevNorm` | Shimmer variability over time. |
| `HNRdBACF_sma3nz_amean` | HNR from 60 ms ACF: 10·log₁₀(ACF at T₀ / (ACF(0) − ACF at T₀)). Lower = more aperiodic / noisy voice → dysphonia, breathiness. |
| `HNRdBACF_sma3nz_stddevNorm` | HNR variability — intermittent vs. sustained aperiodicity. |

**Voice source / spectral balance** (voiced frames)

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `logRelF0-H1-H2_sma3nz_amean` | Log(H1 amplitude / H2 amplitude) | Glottal open quotient proxy. Higher H1-H2 = breathier voice (more open glottis); lower = pressed/modal phonation. |
| `logRelF0-H1-H2_sma3nz_stddevNorm` | H1-H2 variability | Fluctuating phonation mode. |
| `logRelF0-H1-A3_sma3nz_amean` | Log(H1 / amplitude at F3) | Overall voice effort proxy. Higher H1-A3 = softer/breathier; lower = louder/more pressed. |
| `logRelF0-H1-A3_sma3nz_stddevNorm` | H1-A3 variability | |
| `alphaRatioV_sma3nz_amean` | Energy 50–1000 Hz / energy 1–5 kHz (voiced, from LTAS per frame) | Higher = more low-frequency dominance = softer/breathier voice. Lower = more high-frequency energy = louder, harder phonation. |
| `alphaRatioV_sma3nz_stddevNorm` | Alpha ratio variability (voiced) | |
| `alphaRatioUV_sma3nz_amean` | Same ratio (50–1000 Hz / 1–5 kHz) for unvoiced frames | Frication spectral balance; reflects voiceless consonant energy distribution. |
| `hammarbergIndexV_sma3nz_amean` | Peak in 0-2 kHz / peak in 2-5 kHz (voiced) | Higher = low-freq dominant = modal/chest voice. Lower = more high-frequency prominence (pressed, harsh, or falsetto). |
| `hammarbergIndexV_sma3nz_stddevNorm` | Hammarberg index variability | |
| `hammarbergIndexUV_sma3nz_amean` | Hammarberg index (unvoiced frames) | |
| `slopeV0-500_sma3nz_amean` | Spectral slope 0-500 Hz (voiced) | Steepness of low-frequency spectral roll-off; relates to F0 dominance vs. harmonic density. |
| `slopeV0-500_sma3nz_stddevNorm` | Variability of 0-500 Hz slope | |
| `slopeV500-1500_sma3nz_amean` | Spectral slope 500-1500 Hz (voiced) | Mid-frequency spectral shape; influenced by F1-F2 region. |
| `slopeV500-1500_sma3nz_stddevNorm` | Variability of 500-1500 Hz slope | |
| `slopeUV0-500_sma3nz_amean` | Spectral slope 0-500 Hz (unvoiced) | Low-freq slope of frication noise. |
| `slopeUV500-1500_sma3nz_amean` | Spectral slope 500-1500 Hz (unvoiced) | Mid-freq slope of frication. |

**Formants** (voiced frames — `F1/F2/F3frequency/bandwidth/amplitudeLogRelF0_sma3nz_*`)

| Feature | Biomarker interpretation |
|---------|--------------------------|
| `F1/2/3frequency_sma3nz_amean` | Mean F1/F2/F3. Vowel articulation, vocal tract length; see Praat formant guide above. |
| `F1/2/3frequency_sma3nz_stddevNorm` | Formant frequency variability — range of vowel space used; reduced in dysarthria. |
| `F1/2/3bandwidth_sma3nz_amean` | Mean formant bandwidth. Wider = more damping / less resonance sharpness; elevated in hyper- or hypofunctional voices. |
| `F1/2/3bandwidth_sma3nz_stddevNorm` | Bandwidth variability. |
| `F1/2/3amplitudeLogRelF0_sma3nz_amean` | Log amplitude of F1/F2/F3 relative to F0. Formant excitation strength; H1-A1 type measures relate to voice quality. |
| `F1/2/3amplitudeLogRelF0_sma3nz_stddevNorm` | Amplitude ratio variability. |

**Segment-level and energy features**

| Feature | Definition | Biomarker interpretation |
|---------|-----------|--------------------------|
| `loudnessPeaksPerSec` | Loudness peaks per second | Syllabic rate proxy; reduced in slow, dysarthric, or depressed speech. |
| `VoicedSegmentsPerSec` | Voiced segments per second | Speaking rate; fluency indicator. |
| `MeanVoicedSegmentLengthSec` | Mean voiced segment duration | Longer = more sustained phonation (slow speech, singing); shorter = more stop-consonant / fragmented speech. |
| `StddevVoicedSegmentLengthSec` | Std dev of voiced segment duration | Irregular voiced-segment length = variable articulation rhythm. |
| `MeanUnvoicedSegmentLength` | Mean unvoiced gap duration | Longer gaps = more pausing or more voiceless consonants. |
| `StddevUnvoicedSegmentLength` | Std dev of unvoiced gaps | Irregular pausing — cognitive load, fluency disorder. |
| `equivalentSoundLevel_dBp` | Integrated energy over utterance (dBp) | Total vocal output level; overall intensity / loudness. |

### Parsing statistical result files

Feature distribution result files are stored as `results/run_0/feat_dist_all_{feature}.txt`. Each file contains lines starting with `overall:` and `pairwise:` that are Python dict literals — parse them with `ast.literal_eval`, do NOT use string grep on the raw text.

Significance levels:
- **highly significant**: p < 0.001
- **significant**: p < 0.05
- **marginally significant**: p < 0.1
- **not significant**: p >= 0.1

A feature is considered significant if:
- the `overall` Kruskal-Wallis result is not "not significant", OR
- at least one pairwise comparison is not "not significant"

The significance field always contains the literal string "not significant" for non-significant results. Checking with `grep -v "not significant"` on a combined string is unreliable because a single string can contain both "not significant" (for some pairs) and "significant" (for others). Always parse with Python and check each entry individually.

### Fast report generation — use a Bash Python script, not an agent

Delegating file parsing to a subagent is slow (agent startup + many tool calls). Instead, run a single Bash command with an inline Python script that reads and parses all result files at once. Template:

```bash
python3 - <<'EOF'
import ast, glob, os, json

root = "<experiment_root>"          # e.g. experiments/pvqd-dysphonia/results_sustained_mld
targets = ["grade", "roughness", ...]

out = {}
for t in targets:
    out[t] = {}
    pattern = f"{root}/{t}/results/run_0/feat_dist_all_*.txt"
    for path in sorted(glob.glob(pattern)):
        feat = os.path.basename(path).removeprefix("feat_dist_all_").removesuffix(".txt")
        overall, pairwise = None, None
        with open(path) as f:
            for line in f:
                if line.startswith("overall:"):
                    overall = ast.literal_eval(line[len("overall: "):].strip())
                elif line.startswith("pairwise:"):
                    pairwise = ast.literal_eval(line[len("pairwise: "):].strip())
                if overall and pairwise:
                    break
        if overall is None or pairwise is None:
            continue
        o = list(overall.values())[0]
        sig = lambda d: not d["significance"].startswith("not significant")
        is_sig = sig(o) or any(sig(v) for v in pairwise.values())
        if not is_sig:
            continue
        out[t][feat] = {
            "overall_p": o["p_value"], "overall_sig": o["significance"],
            "pairwise": {k: {"p": v["p_value"], "sig": v["significance"]}
                         for k, v in pairwise.items()}
        }

print(json.dumps(out, indent=2))
EOF
```

This replaces the entire agent delegation step. After running it, you have all significant features with p-values and can write the report directly without any further file reads.


## How Claude Should Behave
- Use utility functions from the utils sub-package when possible.
- all pytest calls are allowed
- When the same sequence of statements would appear more than once (e.g. compute stats, then conditionally write two result lines), extract it into a private helper method first and call that helper at each site. Do not inline the same block twice.
