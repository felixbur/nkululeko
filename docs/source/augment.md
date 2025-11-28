# nkululeko.augment

The `nkululeko.augment` module applies audio augmentations (noise, speed, pitch shifts, etc.) to diversify training data and mitigate overfitting.

## Purpose
* Increase dataset variability.
* Enhance robustness to recording conditions.
* Support class balance via synthetic samples.

## Invocation
```bash
python -m nkululeko.augment --config examples/exp_emodb_os_svm.ini
```
(Augmentation options must be specified in `[DATA]` / dedicated augmentation sections; see `ini_file.md`.)

## Common Augmentations
* Additive noise
* Time stretching
* Pitch shifting
* Room impulse simulation

## INI Sketch
```ini
[DATA]
augment = noise,time_stretch
noise.snr = 10
time_stretch.factor = 1.05
```

## Outputs
Augmented features integrated into feature extraction pipeline; logs indicate augmentation steps.

## Tips
* Limit augment layers to avoid excessive runtime.
* Track original vs augmented counts for class balance verification.

## Related
See `balance.md` for algorithmic balancing and `experiment.md` for full flow.
