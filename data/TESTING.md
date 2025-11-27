# Testing the Automated Database README System

This document outlines tests to validate the automated database description system.

## Test 1: Verify All Databases Have descr.yml

```bash
cd data
for dir in */; do
    if [ ! -f "${dir}descr.yml" ]; then
        echo "Missing descr.yml: $dir"
    fi
done
```

Expected: No output (all databases should have descr.yml files)

## Test 2: Validate descr.yml Format

Each descr.yml should have these fields:
- name
- target
- description
- access
- license

```bash
cd data
for dir in */; do
    if [ -f "${dir}descr.yml" ]; then
        # Check if file has required fields
        if ! grep -q "^name:" "${dir}descr.yml"; then
            echo "Missing 'name' in ${dir}descr.yml"
        fi
        # Add similar checks for other fields...
    fi
done
```

## Test 3: Generate README

```bash
cd data
python make_readme.py
```

Expected output: "README.md file created with 66 datasets in . directory."

## Test 4: Add a New Database

```bash
cd data
mkdir -p test-new-db
cat > test-new-db/descr.yml << EOF
name: test-new-db
target: emotion
description: Test database for validation
access: public
license: MIT
EOF

python make_readme.py
grep test-new-db README.md
```

Expected: README should contain the new database entry with 67 total datasets

Cleanup:
```bash
rm -rf test-new-db
python make_readme.py
```

## Test 5: Run from Repository Root

```bash
cd /path/to/nkululeko
python data/make_readme.py
```

Expected: Should work from root directory as well

## Test 6: Verify README Content

Check that README.md:
- Has correct header and documentation
- Contains table with 5 columns (Name, Target, Description, Access, License)
- Lists 66 databases (or current count)
- Databases are sorted alphabetically by name
- Has performance section with image

## Test 7: Verify Git Tracking

```bash
git status data/*/descr.yml
```

Expected: All descr.yml files should be tracked (not ignored)

## Success Criteria

All tests should pass:
- ✅ All 66 database folders have descr.yml files
- ✅ All descr.yml files have correct format
- ✅ README.md is generated correctly
- ✅ New databases can be added easily
- ✅ Script works from both data/ and repository root
- ✅ descr.yml files are tracked by git
