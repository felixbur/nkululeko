# nkululeko.segment

The `nkululeko.segment` module performs voice activity detection (VAD) and segmentation of audio into analyzable chunks.

## Use Cases
* Removing leading/trailing silence.
* Splitting long recordings into uniform segments.
* Improving feature consistency by focusing on voiced regions.

## Invocation
```bash
python -m nkululeko.segment --config examples/exp_emodb_os_svm.ini
```

## Typical INI Options
```ini
[DATA]
segment = vad
segment.min_duration = 0.5
segment.max_duration = 5.0
segment.pad = 0.1
```

## Outputs
Segmented audio references or intermediate files; logs summarizing discarded silence.

## Tips
* Tune `min_duration` to avoid overly short, noisy segments.
* Padding preserves phonetic boundaries when truncating.

## Related
`resample.md` (rate normalization), `experiment.md` (full pipeline).
