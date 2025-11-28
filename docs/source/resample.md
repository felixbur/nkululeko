# nkululeko.resample

The `nkululeko.resample` module checks and normalizes sampling rates across audio files, ensuring feature extractors operate consistently.

## Why Resample?
* Different sampling rates can skew spectral features.
* Some extractors expect uniform sampling frequency.

## Invocation
```bash
python -m nkululeko.resample --config examples/exp_emodb_os_svm.ini
```

## Behavior
* Scans dataset paths from `[DATA]`.
* Reports any deviations and optionally writes resampled copies.

## INI Options (Illustrative)
```ini
[DATA]
target_rate = 16000
resample_action = warn   # or fix
```

## Outputs
Log report of sampling rates; optionally new audio files in structured folders if fixing.

## Tips
* Prefer fixing before large feature extraction runs.
* Keep an original backup to avoid irreversible transformations.

## Related
`segment.md` (VAD / segmentation), `experiment.md` (pipeline integration).
