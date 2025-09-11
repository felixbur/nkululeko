# Nkuluflag Module (Legacy)

> **Note**: The `nkuluflag` module has been replaced by the more powerful `flags` module. Please see the [Flags Module Tutorial](flags.md) for the current recommended approach.

## Legacy Usage: 

  ```bash
  $ python -m nkululeko.nkuluflag [-h] [--config CONFIG] [--data [DATA ...]] [--label [LABEL ...]] [--tuning_params [TUNING_PARAMS ...]] [--layers [LAYERS ...]] [--model MODEL] [--feat FEAT] [--set SET] [--with_os WITH_OS] [--target TARGET] [--epochs EPOCHS] [--runs RUNS] [--learning_rate LEARNING_RATE] [--drop DROP]
  ```

## Migration to Flags Module

The new `flags` module provides several advantages over the legacy `nkuluflag`:

1. **Configuration-based approach**: Define parameter combinations in your INI file instead of command line
2. **Better performance**: Optimized feature extraction that runs only once
3. **More flexible**: Support for custom parameters and different data types
4. **Better error handling**: Individual experiment failures don't stop the entire process
5. **Clear output**: Comprehensive summary with best configuration identification

### Migration Example

**Old nkuluflag approach:**
```bash
python -m nkululeko.nkuluflag --config base.ini --model xgb svm --feat os praat
```

**New flags approach:**
1. Add a `[FLAGS]` section to your INI file:
```ini
[FLAGS]
models = ['xgb', 'svm']
features = ['os', 'praat']
```

2. Run with the flags module:
```bash
python -m nkululeko.flags --config base.ini
```

For complete documentation and examples, please refer to the [Flags Module Tutorial](flags.md).