# Database Description System - Implementation Summary

## Overview

The Nkululeko data directory now uses a decentralized approach for managing database metadata. Each database folder contains its own `descr.yml` file, and the main `data/README.md` is automatically generated from these individual files.

## What Changed

### Before
- Single central `data/descr.yml` file containing all database information
- Difficult to maintain as number of databases grew
- Easy to miss updating the central file when adding new databases

### After
- Each database folder has its own `descr.yml` file
- `make_readme.py` scans all database folders and generates the table automatically
- Easy to add new databases - just create a folder with `descr.yml`
- Scalable and maintainable

## File Structure

```
data/
├── README.md                    # Auto-generated, DO NOT edit manually
├── descr.yml                    # Old central file (kept for reference)
├── make_readme.py               # README generator script
├── migrate_descr.py             # Migration utility (one-time use)
├── DESCR_FORMAT.md              # Documentation for descr.yml format
├── TESTING.md                   # Validation test procedures
├── emodb/
│   ├── descr.yml               # Database metadata
│   ├── README.md               # Database-specific documentation
│   └── ...
├── polish/
│   ├── descr.yml
│   └── ...
└── ... (66 total database folders)
```

## descr.yml Format

Each database folder should contain a `descr.yml` file with this structure:

```yaml
name: database-name
target: emotion                 # or multiple: "emotion,speaker"
description: Brief description
access: public                  # or "restricted" or "private"
license: CC BY 4.0             # or other license
```

## Usage

### Regenerating the README

From the `data/` directory:
```bash
cd data
python make_readme.py
```

From the repository root:
```bash
python data/make_readme.py
```

### Adding a New Database

1. Create a new folder in `data/`:
   ```bash
   mkdir data/my-new-database
   ```

2. Create `descr.yml`:
   ```bash
   cat > data/my-new-database/descr.yml << EOF
   name: my-new-database
   target: emotion
   description: My new emotion database
   access: public
   license: MIT
   EOF
   ```

3. Regenerate README:
   ```bash
   python data/make_readme.py
   ```

4. Commit the changes:
   ```bash
   git add data/my-new-database/descr.yml data/README.md
   git commit -m "Add my-new-database"
   ```

## Migration

The migration from the central `descr.yml` to individual files was done using the `migrate_descr.py` script:
- 57 databases were migrated from the central file
- 9 new databases were added with template files
- Total: 66 databases now have individual `descr.yml` files

## Validation

Run these checks to ensure the system is working correctly:

1. All databases have `descr.yml`:
   ```bash
   cd data && for dir in */; do [ ! -f "${dir}descr.yml" ] && echo "Missing: $dir"; done
   ```

2. README is up to date:
   ```bash
   cd data && python make_readme.py && git diff README.md
   ```
   (Should show no diff if already up to date)

3. Count matches:
   ```bash
   cd data && grep "contains information about" README.md
   ```
   (Should show 66 datasets)

## Benefits

1. **Scalability**: Easy to add new databases without touching a central file
2. **Maintainability**: Database metadata lives with the database itself
3. **Automation**: README is always generated from source of truth
4. **Git-friendly**: Changes to one database don't affect others in version control
5. **Documentation**: Each database folder is self-documenting
6. **Consistency**: Enforced structure through YAML validation

## Dependencies

- Python 3.x
- PyYAML (for reading YAML files)
- mdutils (for generating Markdown tables)

Install with:
```bash
pip install pyyaml mdutils
```

## Future Enhancements

Potential improvements for the future:
- Validate descr.yml format with schema validation
- Add more metadata fields (e.g., citation, DOI, download URL)
- Generate additional documentation from descr.yml
- Create a web interface for browsing databases
- Automate README generation in CI/CD pipeline

## Files Modified/Created

### Modified
- `.gitignore` - Added exception to track descr.yml files
- `data/README.md` - Now auto-generated
- `data/make_readme.py` - Updated to read from individual descr.yml files

### Created
- `data/*/descr.yml` - 66 individual database description files
- `data/DESCR_FORMAT.md` - Format documentation
- `data/migrate_descr.py` - Migration utility
- `data/TESTING.md` - Validation tests
- `data/SUMMARY.md` - This file

## Contact

For questions or issues related to the database description system, please open an issue on the GitHub repository.
