# Quick Start Guide for Database Contributors

This guide shows you how to add a new database to Nkululeko using the automated description system.

## Step-by-Step: Adding a New Database

### 1. Create Database Folder

```bash
cd data
mkdir my-emotion-dataset
cd my-emotion-dataset
```

### 2. Create descr.yml

Create a file named `descr.yml` with the following content:

```yaml
name: my-emotion-dataset
target: emotion
description: English speech emotion recognition dataset
access: public
license: CC BY 4.0
```

**Field Descriptions:**
- `name`: Folder name (should match the directory name)
- `target`: What the dataset is used for (e.g., emotion, age, gender, speaker, VAD)
  - Multiple targets: `emotion,speaker` or `valence,arousal,dominance`
- `description`: Brief description (language, special features, etc.)
- `access`: 
  - `public`: Freely downloadable without restrictions
  - `restricted`: Requires registration or agreement
  - `private`: Not publicly available
- `license`: Dataset license (e.g., CC BY 4.0, MIT, unknown)

### 3. Add Other Files (Optional)

```bash
# Create a README for your database
cat > README.md << EOF
# My Emotion Dataset

Description of how to download and process this dataset.

## Download

...

## Processing

...
EOF

# Add processing scripts, configuration files, etc.
```

### 4. Update the Main README

```bash
cd ..  # Go back to data/ directory
python make_readme.py
```

You should see:
```
README.md file created with 67 datasets in . directory.
```

### 5. Verify the Changes

```bash
# Check that your database appears in the table
grep my-emotion-dataset README.md
```

You should see a line like:
```
|my-emotion-dataset|emotion|English speech emotion recognition dataset|public|CC BY 4.0|
```

### 6. Commit Your Changes

```bash
cd ..  # Go back to repository root
git add data/my-emotion-dataset/descr.yml data/README.md
git commit -m "Add my-emotion-dataset database"
git push
```

**Note**: The CI workflow will automatically validate:
- Your `descr.yml` file has all required fields
- The `README.md` is properly updated
- All database folders have valid metadata

See `CI_VALIDATION.md` for details on the automated checks.

## Examples

### Example 1: Simple Emotion Database

```yaml
name: sample-emotion-db
target: emotion
description: German emotional speech
access: public
license: MIT
```

### Example 2: Multi-Target Database

```yaml
name: multimodal-db
target: emotion,age,gender
description: Multimodal speech dataset with multiple labels
access: restricted
license: Custom academic/research use
```

### Example 3: VAD Database

```yaml
name: vad-corpus
target: valence,arousal,dominance
description: Continuous emotion annotation dataset
access: public
license: CC BY-NC 4.0
```

### Example 4: Clinical Dataset

```yaml
name: parkinsons-speech
target: Parkinson's disease
description: Spanish speech for Parkinson's detection
access: restricted
license: for research
```

## Common Mistakes to Avoid

❌ **Don't** manually edit `data/README.md` - it's auto-generated!

❌ **Don't** use uppercase in folder/database names - use lowercase with hyphens

❌ **Don't** forget to run `make_readme.py` after adding your database

❌ **Don't** forget to commit both `descr.yml` and the updated `README.md`

✅ **Do** use consistent naming between folder and descr.yml `name` field

✅ **Do** provide accurate license information

✅ **Do** test that README generation works before committing

## Troubleshooting

**Q: My database doesn't appear in README.md**

A: Make sure:
1. The folder is directly under `data/` (not nested)
2. The file is named exactly `descr.yml` (not `descr.yaml`)
3. The YAML syntax is valid
4. You ran `make_readme.py` after creating the file

**Q: I get a YAML parsing error**

A: Check your `descr.yml` syntax:
- Use `name:` not `- name:`
- No tabs (use spaces for indentation)
- Quotes around values with special characters

**Q: The order in README is wrong**

A: The README is automatically sorted alphabetically by database name. You can't control the order.

**Q: I need to update database information**

A: Just edit the `descr.yml` file and run `make_readme.py` again.

## Need Help?

- See `DESCR_FORMAT.md` for detailed format documentation
- See `TESTING.md` for validation procedures
- See `SUMMARY.md` for complete system overview
- Open an issue on GitHub for questions

## Quick Reference

```bash
# Create database
mkdir data/my-db
cat > data/my-db/descr.yml << EOF
name: my-db
target: emotion
description: My database
access: public
license: MIT
EOF

# Update README
python data/make_readme.py

# Commit
git add data/my-db/descr.yml data/README.md
git commit -m "Add my-db"
```

---

**Happy Contributing! 🎉**
