# Domain-adversarial training for cross-corpus audio deepfake detection (AASIST, HAD held out)

This is the write-up behind `[MODEL] type = aasist` and `[MODEL] dann_columns`.
It compares bare AASIST with AASIST trained against *which corpus* and
*which language* a clip comes from (DANN), when the test corpus is never seen
in training. The metric is EER on the held-out test corpus unless stated
otherwise.

`configs/` has the three configs needed to repeat the headline comparison:

| config | what it trains | runs |
|---|---|---|
| `exp_aasist_had_baseline.ini` | bare AASIST | 5 |
| `exp_aasist_had_dann_source_db.ini` | AASIST + DANN against the source database | 8 |
| `exp_aasist_had_dann_language.ini` | AASIST + DANN against the clip language | 8 |

**Where the numbers come from.** All numbers below were measured on the
author's fork, before the review changes that went into the merged AASIST
(#460) and DANN (#465) code. The merged code has not been re-run on the full
5/8-run set, so a repeat may differ in detail. The fork also had RawBoost,
SAM and domain-balanced sampling; those are not in upstream (see §6).

## 1. Setup

- **Model**: AASIST (Jung et al., ICASSP 2022) on top of the default SSL
  frontend `facebook/wav2vec2-xls-r-300m`, fine-tuned end to end
  (`freeze_ssl_frontend = False`, last layer only). Freezing the frontend was
  tried once and dropped: 26.8% dev EER after 8 epochs, against about 4% for
  the unfrozen model at the same point.
- **Folds**: leave-one-dataset-out. Train on `itw`, `la`, `cvoicefake` and
  `habla`; early-stop on `for2sec` (dev); test on `had` (HAD).
- **Data budget**: a stratified sample of 2500 real + 2500 fake clips from
  each training corpus (20,000 training rows), and 1000 + 1000 for dev and
  for test (seed 42). The full pool (over 200k rows) made one epoch take
  hours.
- **Training**: `lr = 1e-5`, Adam, batch size 16, `patience = 5`, at most 50
  epochs, cross-entropy. The training set is balanced, so no class weights.
  Everything except the `dann_*` keys is identical between the configs.
- **Seeds**: `runs = N` trains N models from different random
  initialisations (nkululeko fixes no seed unless `[MODEL] random_seed` is
  set, and then all runs are identical). With `save = False` and
  `traindevtest = True` only the best dev checkpoint of a run is kept.

### Building the CSVs

The configs read `./data/multidb_ood/_aasist_subsample_2500_lang/<db>.csv`,
with the audio under `./data/...` (relative to the directory you run
nkululeko from; a relative `file` entry is resolved against the database's
`audio_path`). The CSVs are not part of this repository. Each has the columns
`file` and `label` (`fake` / `real`); all but `had` also have `speaker`.

The `language` column is only used by the `language` DANN config:

| database | `language` |
|---|---|
| `itw`, `la` | `en` |
| `habla` | `es` |
| `cvoicefake` | its own per-row value (`it`, `de`, `zh-CN`, `fr`, `en`) |

The language distribution of the 20,000 training rows is not a recoding of
`source_db` (4 values): `en` 10,931 (54.7%), `es` 5,000 (25%), `de` 1,514,
`zh-CN` 979, `it` 814, `fr` 762. `dev` and `test` need no `language` column,
because DANN only reads domain labels from the training split. The sampling
script used for the report was a one-off and was not kept, so CSVs you rebuild
the same way contain the same *kind* of sample, not necessarily the same
rows.

## 2. A calibration finding that qualifies every EER number

At the model's own decision threshold (not the EER-optimal one), bare AASIST
on `had` predicts "real" for almost every held-out clip. The single
reference run (7.95% EER) had UAR 53.6% and a fake-class specificity of 7.3%;
the other AASIST configs on this fold had fake-class specificity as low as
1.5-12% and UAR of about 50-56%. EER looks fine because it is threshold-free
(the scores still rank clips correctly), but a system deployed with AASIST's
native threshold would catch almost no fakes in this domain. Read every EER
below with that in mind.

## 3. Why single runs cannot be trusted

The first bare-AASIST run on `had` gave 7.95% EER. Reseeded, three more
runs of the same config gave:

| bare AASIST, `had` | run 1 | run 2 | run 3 | mean | std |
|---|---|---|---|---|---|
| EER | 14.90% | 27.15% | 18.80% | 20.28% | 5.11pp |

The first run was below the minimum of the three reseeded runs. With five
runs the bare baseline is **17.30% +- 6.04pp**. Run-to-run std is about a
third of the mean, so several single-run conclusions flipped sign once
reseeded. Every comparison below therefore reports its number of runs and a
significance test.

**The test.** p-values are exact permutation tests on the difference of means
(all `C(2n, n)` equal-size splits of the pooled runs, no distributional
assumption). Welch's t-test is shown next to it only to flag disagreement; at
n=3 it can report a nominally significant p-value that the exact test
contradicts. The smallest possible exact p-value is limited by n: 1/20 = 0.05
at n=3, 1/252 = 0.004 at n=5, 1/12,870 = 0.00008 at n=8 (per arm). So n=3 can
hardly ever give a result to believe, whatever the true effect.

## 4. DANN: source database vs. language

Two DANN configs, identical except for `dann_columns`, run for 8 seeds each
(the only cells pushed this far):

| config | n | mean EER | std |
|---|---|---|---|
| bare AASIST (reference) | 5 | 17.30% | 6.04pp |
| DANN, `dann_columns = ['source_db']` | 8 | 16.75% | 6.61pp |
| DANN, `dann_columns = ['language']` | 8 | 26.14% | 6.79pp |

`language` minus `source_db`: **+9.39pp EER** (Welch p = 0.014, exact
permutation p = 0.016). Welch and the exact test agree closely here, unlike
in every other comparison of this study, which is some evidence the result is
not a small-sample artifact.

What this does and does not say:

- Training against **language** identity gives a clearly worse held-out EER
  than training against **database** identity.
- It does *not* show that DANN on `source_db` helps: 16.75% vs 17.30% for the
  bare model is well inside the run-to-run noise (this pair was not
  tested).
- A two-head run (`['source_db', 'language']`) with only 3 seeds gave 25.75%
  +- 12.57pp, close to the language-only arm. That points at the language axis
  rather than at stacking two adversarial heads, but at n=3 it is not
  conclusive (exact p = 0.21 against the `source_db` arm).

## 5. Notes for repeating the runs

- Set `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) for unattended
  queues: one run hung for over a day on a stale HuggingFace Hub socket.
- An AASIST checkpoint with the default frontend is about 1.2 GB, and
  `save = True` writes one per epoch; keep `save = False` (as in the configs)
  unless you need the models.
- Compare configs only at equal data and equal `runs`; the table in §4 mixes
  5 and 8 runs.

## 6. Other techniques (not in upstream)

On the same fold the fork also tested RawBoost augmentation, SAM,
domain-balanced batch sampling and a second held-out fold (`itw`). With 3-5
runs per arm none of them differed significantly from the bare baseline, in
either direction. They could still be underpowered false negatives. Because
their code is not part of upstream nkululeko, neither are their configs or
tables; they are in <https://github.com/bagustris/nkululeko/pull/66>.

## 7. Limitations

- Only one held-out fold (`had`) is used for the DANN comparison; the
  `itw` fold was used for the other techniques only.
- The merged implementation has not been re-run on the full run set (see the
  top of this report).
- The bare baseline has 5 runs and the DANN arms 8; bare vs. DANN was not
  tested.
- Calibration (§2) was checked on the single reference run and a few other
  configs, not on every run.
