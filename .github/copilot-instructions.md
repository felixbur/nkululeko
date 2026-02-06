# Copilot Instructions for Nkululeko

## Big picture
- Nkululeko runs ML experiments end-to-end from INI configs: dataset loading → feature extraction → model training → reporting. Orchestration is centered in nkululeko/experiment.py and invoked by CLI modules in nkululeko/*.py.
- Experiments are configured via INI files (see ini_file.md and examples/*.ini). Key sections: [EXP], [DATA], [FEATS], [MODEL], [EXPL].
- Datasets are defined under data/<dataset>/ with metadata and configs; runnable examples live in examples/ and meta/demos/.

## Key entry points (all accept --config <file.ini>)
- nkululeko.nkululeko (main runner)
- nkululeko.ensemble (late fusion), nkululeko.multidb (multi-database)
- nkululeko.demo, test, explore, augment, optim, predict, resample, segment

## Developer workflows
- Install dev deps: pip install -r requirements-test.txt && pip install -e .
- Tests: python tests/test_install.py (creates build/ venv) or python -m unittest tests/test_modules.py
- Format/lint: black nkululeko/ --exclude nkululeko/constants.py; isort --profile black nkululeko/; ruff check --fix --output-format=full nkululeko
- Docs: cd docs && uv pip install -r requirements.txt && make html (builds docs/build/html)

## Project-specific conventions
- Do not reformat nkululeko/constants.py (VERSION string is intentionally excluded from black).
- Most behavior is driven by INI keys; when adding options, update ini_file.md and provide an example config in examples/.
- Experiments write outputs to exp_*/ and results/; long runs and large artifacts are expected.

## External dependencies & gotchas
- Optional ML backends: PyTorch, TensorFlow, Spotlight (Spotlight unsupported on Python 3.13+).
- For Ubuntu audio deps: sox and libportaudio2 are required for some features.
- CI includes a heavy integration run (py310-aud-csv.yml) that downloads datasets and expects “DONE” in output.
