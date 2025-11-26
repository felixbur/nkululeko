# Database Description Files

## Overview

Each database folder in the `data/` directory should contain a `descr.yml` file with metadata about the dataset. The `make_readme.py` script automatically scans all database folders, reads their `descr.yml` files, and generates the main `data/README.md` file with a comprehensive table of all datasets.

## Format

Each `descr.yml` file should contain the following fields:

```yaml
name: database-folder-name
target: emotion  # or other target(s) like "emotion,VAD", "speaker", "age,gender", etc.
description: Brief description of the dataset
access: public  # or "restricted" or "private"
license: CC BY 4.0  # or other license type, use "unknown" if not specified
```

### Field Descriptions

- **name**: The name of the database (should match the folder name)
- **target**: The target variable(s) that the dataset is used for (e.g., emotion, age, gender, VAD, speaker, etc.). Multiple targets can be comma-separated.
- **description**: A brief description of the dataset (language, special characteristics, etc.)
- **access**: Accessibility level:
  - `public`: Publicly available without restrictions
  - `restricted`: Publicly available but requires registration or agreement
  - `private`: Not publicly available, requires private access from dataset owners
- **license**: The license under which the dataset is distributed

## Adding a New Database

When adding a new database to the `data/` directory:

1. Create a new folder with the database name
2. Create a `descr.yml` file in the folder with the required metadata
3. Add any other necessary files (README, processing scripts, etc.)
4. Run `python make_readme.py` from the `data/` directory to regenerate the main README

## Regenerating the README

To update the `data/README.md` file after adding or modifying databases:

```bash
cd data
python make_readme.py
```

This will:
- Scan all subdirectories in the `data/` folder
- Read each `descr.yml` file
- Generate a sorted table with all database information
- Update the dataset count

## Migration from Central descr.yml

The old system used a single central `data/descr.yml` file containing all database information. This has been migrated to individual `descr.yml` files in each database folder for better maintainability and scalability.

The migration script `migrate_descr.py` was used to split the central file into individual files. The central `descr.yml` is kept for reference but is no longer used by `make_readme.py`.

## Example

Example `descr.yml` for the emoDB database:

```yaml
name: emodb
target: emotion
description: German
access: public
license: CC BY 4.0
```
