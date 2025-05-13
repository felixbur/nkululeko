Proposed test directory structure (by Devin)

```bash
nkululeko/  
└── tests/  
    ├── conftest.py           # Shared pytest fixtures  
    ├── test_data/            # Small test datasets  
    ├── test_configs/         # Test configuration files  
    ├── unit/                 # Unit tests  
    │   ├── test_experiment.py  
    │   ├── test_models.py  
    │   ├── test_features.py  
    │   └── ...  
    ├── integration/          # Integration tests  
    │   ├── test_pipeline.py  
    │   ├── test_explore.py  
    │   └── ...  
    └── cli/                  # CLI tests
```