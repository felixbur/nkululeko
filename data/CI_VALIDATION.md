# CI Workflow for Database Metadata Validation

This document describes the automated CI workflow that validates database metadata and ensures the README is kept up to date.

## Workflow: validate_database_metadata.yml

### Triggers

The workflow runs automatically on:
- **Push events** when changes are made to:
  - `data/*/descr.yml` files
  - `data/make_readme.py` script
  
- **Pull request events** for the same paths

### What It Does

The workflow performs three main validation steps:

#### 1. Validate descr.yml Files Exist

Checks that every database folder in `data/` has a `descr.yml` file.

If any database folder is missing a `descr.yml` file, the workflow will:
- ❌ Fail the CI check
- List all folders missing the file
- Provide instructions on how to create one

#### 2. Validate descr.yml Format

Verifies that each `descr.yml` file contains all required fields:
- `name`
- `target`
- `description`
- `access`
- `license`

If any file is missing required fields, the workflow will:
- ❌ Fail the CI check
- List which files are invalid and which fields are missing
- Provide format requirements

#### 3. Validate README.md is Up to Date

Regenerates the `data/README.md` file and compares it with the committed version.

If the README is outdated, the workflow will:
- ❌ Fail the CI check
- Show a diff of what changed
- Provide instructions to update the README

## For Contributors

### When Adding a New Database

When you add a new database folder, you **must**:

1. Create a `descr.yml` file in the database folder:
   ```yaml
   name: my-new-db
   target: emotion
   description: Brief description
   access: public
   license: MIT
   ```

2. Regenerate the README:
   ```bash
   cd data
   python make_readme.py
   ```

3. Commit both files:
   ```bash
   git add data/my-new-db/descr.yml data/README.md
   git commit -m "Add my-new-db database"
   ```

### When Updating Database Metadata

When you update a `descr.yml` file, you **must**:

1. Make your changes to the `descr.yml` file

2. Regenerate the README:
   ```bash
   cd data
   python make_readme.py
   ```

3. Commit both files:
   ```bash
   git add data/my-db/descr.yml data/README.md
   git commit -m "Update my-db metadata"
   ```

### CI Failure Scenarios

#### Scenario 1: Missing descr.yml

```
❌ Missing descr.yml in my-new-db/

ERROR: Some database folders are missing descr.yml files
Please create a descr.yml file in each database folder
```

**Solution**: Create the missing `descr.yml` file with all required fields.

#### Scenario 2: Invalid descr.yml Format

```
❌ Missing required field 'license' in my-db/descr.yml

ERROR: Some descr.yml files have invalid format
Required fields: name, target, description, access, license
```

**Solution**: Add the missing field to your `descr.yml` file.

#### Scenario 3: Outdated README.md

```
⚠️  README.md needs to be updated

ERROR: README.md is not up to date with the current descr.yml files

Please run the following command to update README.md:
  cd data && python make_readme.py
```

**Solution**: Run `make_readme.py` and commit the updated README.

## Testing Locally

Before pushing, you can test the validation locally:

### Test 1: Validate all descr.yml files exist

```bash
cd data
for dir in */; do
  if [ ! -f "${dir}descr.yml" ]; then
    echo "Missing: $dir"
  fi
done
```

### Test 2: Validate descr.yml format

```bash
cd data
for dir in */; do
  if [ -f "${dir}descr.yml" ]; then
    for field in name target description access license; do
      if ! grep -q "^${field}:" "${dir}descr.yml"; then
        echo "Missing '$field' in ${dir}descr.yml"
      fi
    done
  fi
done
```

### Test 3: Check if README is up to date

```bash
cd data
python make_readme.py
git diff README.md
# If there's output, README needs to be committed
```

## Workflow Configuration

The workflow file is located at:
```
.github/workflows/validate_database_metadata.yml
```

### Dependencies

The workflow requires:
- Python 3.12
- `pyyaml` package
- `mdutils` package

These are automatically installed by the workflow.

### Customization

To modify the validation rules, edit the workflow file. The validation logic is implemented as bash scripts in the workflow steps.

## Benefits

This CI workflow ensures:
- ✅ All databases have proper metadata files
- ✅ Metadata files follow the required format
- ✅ The main README is always synchronized with database metadata
- ✅ Contributors get immediate feedback on issues
- ✅ Prevents incomplete database additions from being merged

## Troubleshooting

**Q: The CI fails but my descr.yml looks correct**

A: Check for:
- Typos in field names (e.g., `descr` instead of `description`)
- Tabs instead of spaces in YAML
- Missing colons after field names
- Extra or missing whitespace

**Q: I updated descr.yml but forgot to update README**

A: Simply run `python data/make_readme.py` and commit the updated README.

**Q: The workflow doesn't trigger**

A: The workflow only triggers when files in `data/*/descr.yml` or `data/make_readme.py` are modified. Other changes won't trigger it.

## Related Documentation

- `DESCR_FORMAT.md` - Format specification for descr.yml
- `CONTRIBUTING_DATABASES.md` - Quick start guide for adding databases
- `TESTING.md` - Manual validation procedures
- `SUMMARY.md` - Complete system overview
