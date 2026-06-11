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

### MldSust feature interpretation (acoustic biomarker guide)

**Summary statistic suffixes** (MldSust uses *robust* statistics)

| Suffix | Definition |
|--------|-----------|
| `_median` | median of the feature time series |
| `_iqr` | interquartile range (p75−p25) |
| `_q_var` | median RMS from median |
| `_q_skewness` | ((p90−p50)−(p50−p10)) / (p90−p10) — tail asymmetry |
| `_q_range5` | p95−p5 — near-total range |
| `_diff` | mean second half minus mean first half (normalised to [0,1]) |
| `_slope` | 1st-order linear fit coefficient (time and feature normalised to [0,1]) |
| `_start` | y[0] − median(y) — deviation at onset |
| `_peak` | max(y) − median(y) — deviation at maximum |
| `_end` | y[−1] − median(y) — deviation at offset |

**Feature groups — definition and biomarker interpretation**

| Feature | Definition (from docs) | Acoustic biomarker interpretation |
|---------|------------------------|-----------------------------------|
| `vq_cpp_*` | Summary stat of cepstral peak prominence | Strength of the harmonic peak in the cepstrum. Lower CPP = more breathy/aperiodic voice. Single strongest marker of overall dysphonia. Median → average quality; peak → best-quality moment (relevant for strain). |
| `shape_en/f0_y_*` | Summary stat over the raw energy/F0 time series | Energy: overall loudness level and contour. Slope < 0 = fading voice (breath/fatigue). Median = average vocal effort (key for asthenia). F0: pitch level and shape. |
| `shape_en/f0_dlt_*` | Summary stat over frame-to-frame deltas of energy/F0 | Perturbation / micro-instability. F0 delta → macro-jitter proxy, captures roughness and strain. Energy delta → shimmer-like amplitude instability. |
| `shape_en/f0_mae` | Mean absolute deviation between time series and its midline | Deviation from the register midline — how far the contour wanders from its own trend. High values = erratic, unsustained phonation. |
| `shape_en/f0_d_start` | Segment-initial distance to midline | How far the energy/F0 is from its midline at the very start — captures phonation onset irregularity. |
| `shape_en/f0_d_end` | Segment-final distance to midline | Distance from midline at the end of the utterance — captures whether energy/pitch deviates at offset (voice runout, breath support failure). |
| `f0en_rms` | RMS deviation between centred+scaled F0 and energy contour | Measures **decoupling** of pitch and energy trajectories. Healthy voices show coordinated F0 and energy; high values indicate the two are out of sync → roughness marker. |
| `pe_en_rms_init` | Energy LPC residual RMSD of first 200 ms | LPC residual = unexplained energy after prediction; high RMSD = irregular energy, especially at onset. Captures rough or strained voice attack. |
| `pe_en_rms_final` | Energy LPC residual RMSD of final 200 ms | Same as above but at offset — irregular energy run-out. |
| `spec_spread_frm_*` | Summary stat of spectral variation over the full sustained sound (inverse spectral stability) | Higher values = less stable spectrum = more noise/aperiodicity. Key breathiness marker; also rises with overall grade. |
| `spec_flux_frm_*` | Summary stat of spectral flux over the full sustained sound | Frame-to-frame spectral change. High range (q_range5) = unstable resonance patterns → asthenia, breathiness. |
| `reg_rng_bl` | F0 register range mean divided by baseline mean | Normalised pitch modulation depth: how large the F0 range is relative to the speaker's own baseline pitch. Asthenic voices show reduced or irregular pitch modulation. |

**Cross-target interpretation patterns (PVQD sustained vowel)**
- **Grade** (overall severity): dominated by CPP median (aperiodicity), F0-delta median (pitch instability), and energy slope (fading voice).
- **Roughness**: driven by F0-energy decoupling (`f0en_rms`), F0-delta spread (`shape_f0_dlt_iqr`) — pitch perturbation and irregular periodicity.
- **Asthenia**: low energy (`shape_en_y_median`), reduced normalised pitch range (`reg_rng_bl`), reduced CPP — all consistent with weak, under-powered phonation.
- **Breathiness**: CPP drop plus increased spectral spread variability (`spec_spread_frm_iqr`) — air leakage raises spectral noise and destabilises the spectrum.
- **Strain**: broadest feature set; F0-delta and CPP-peak rather than median reflect hyperfunction (high effort, intermittent quality loss) and effortful pitch modulation.

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
