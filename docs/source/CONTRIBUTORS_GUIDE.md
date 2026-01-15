# Contributor's Guide for Nkululeko Documentation (Case study: Activation Functions Tutorial)

## Overview
This guide helps contributors understand, maintain, and extend the Nkululeko's tutorial documentation.

## Documentation Structure

### File Hierarchy
```
tutorials/
├── tut_activation_functions.md              [MAIN] Full tutorial
├── ACTIVATION_FUNCTIONS_QUICKREF.md         [REF] Quick reference
├── ACTIVATION_FUNCTIONS_DOCUMENTATION_INDEX [INFO] Index file
├── README.md                                [NAV] Directory index
├── tut_activation_leaky_relu.ini            [EXAMPLE] Working config
├── tut_activation_compare.ini               [TEMPLATE] Testing template
├── compare_activations.sh                   [TOOL] Automation
└── activation_functions_diagram.txt         [VISUAL] Diagrams
```

## Maintenance Tasks

### Regular Updates

#### When to Update
- [ ] New activation function added to codebase
- [ ] Performance benchmarks change significantly
- [ ] Bug fixes in activation implementation
- [ ] New best practices discovered
- [ ] User feedback indicates confusion

#### What to Update
1. **Code examples** - Keep in sync with actual implementation
2. **Performance numbers** - Update benchmark results
3. **Troubleshooting** - Add new common issues
4. **Links** - Verify all cross-references work

### Testing Documentation

Before committing changes:

```bash
# 1. Test all INI configurations
python -m nkululeko.nkululeko --config tutorials/tut_activation_leaky_relu.ini
python -m nkululeko.nkululeko --config tutorials/tut_activation_compare.ini

# 2. Test automation script
./tutorials/compare_activations.sh

# 3. Verify markdown rendering (if using a markdown viewer)
cat tutorials/tut_activation_functions.md

# 4. Check all links
grep -r "](.*)" tutorials/*.md | grep -v "http" # Local links
grep -r "http" tutorials/*.md                    # External links
```

### Version Control

#### File Versioning
Each major documentation file includes:
- **Version**: Incremented on significant changes
- **Date**: Last update date
- **Compatibility**: Nkululeko version required

Example header:
```markdown
# Title
**Version**: 1.1
**Last Updated**: 2026-01-15  
**Requires**: nkululeko >= 1.1.2
```

## Adding New Content

### Adding a New Activation Function

When a new activation (e.g., GELU) is added to the codebase:

1. **Update main tutorial** (`tut_activation_functions.md`):
   ```markdown
   #### GELU (Gaussian Error Linear Unit)
   ```
   f(x) = x * Φ(x)  where Φ is the standard Gaussian CDF
   ```
   - **Advantages**: Smooth, used in transformers
   - **Use when**: Working with transformer-based models
   ```

2. **Update quick reference** (`ACTIVATION_FUNCTIONS_QUICKREF.md`):
   Add to table and decision tree

3. **Create example config**:
   ```ini
   [MODEL]
   type = mlp
   activation = gelu
   ```

4. **Update comparison script** (`compare_activations.sh`):
   ```bash
   activations=("relu" "leaky_relu" "tanh" "sigmoid" "gelu")
   ```

5. **Test thoroughly**:
   Run all examples and verify output

### Adding a New Example

Template for new examples:

```ini
# File: tut_activation_<name>.ini
# Description: <Brief description>

[EXP]
root = ./tutorials/results/
name = tut_activation_<name>
runs = 1  # Use 1 for quick tests
epochs = 10  # Keep low for tutorials

[DATA]
# Use common datasets (emodb, etc.)
databases = ['emodb']
emodb = ./data/emodb/emodb
target = emotion

[FEATS]
type = ['os']
scale = standard  # Always include scaling

[MODEL]
type = mlp
layers = [64, 32]  # Keep simple for tutorials
activation = <your_activation>
drop = 0.2
patience = 5

# Add comments explaining choices
```

## Style Guide

### Markdown Formatting

1. **Headers**: Use ATX-style (`#` syntax)
2. **Code blocks**: Always specify language
   ```ini
   [MODEL]
   type = mlp
   ```
3. **Tables**: Align columns with pipes
4. **Lists**: Use consistent bullet style
5. **Emphasis**: 
   - **Bold** for important terms
   - *Italic* for emphasis
   - `Code` for parameters/values

### Writing Style

- **Be concise**: Tutorial users want quick answers
- **Be practical**: Focus on actionable advice
- **Be accurate**: Test all code examples
- **Be helpful**: Anticipate user questions

### Code Examples

Rules for configuration examples:
- ✅ Always include comments
- ✅ Use realistic values
- ✅ Test before committing
- ✅ Keep examples simple
- ❌ Don't use placeholder values
- ❌ Don't assume prior knowledge

## Documentation Principles

### 1. Progressive Disclosure
- Start simple, add complexity gradually
- Beginners: Basic examples first
- Advanced: Deep dives later

### 2. Learning Styles
Support different learning approaches:
- **Visual**: Diagrams and flowcharts
- **Practical**: Working examples
- **Theoretical**: Explanations and math
- **Reference**: Quick lookup tables

### 3. Completeness
Every tutorial should include:
- What it is
- Why it matters
- How to use it
- When to use it
- Examples
- Troubleshooting

## Common Pitfalls

### What to Avoid

1. **Outdated examples**: Test regularly
2. **Missing error handling**: Show failure cases
3. **Unclear prerequisites**: State requirements upfront
4. **Broken links**: Verify all URLs
5. **Untested code**: Run every example
6. **Platform assumptions**: Note OS-specific details

### Quality Checklist

Before submitting:
- [ ] All code examples tested
- [ ] Links verified
- [ ] Spelling checked
- [ ] Consistent formatting
- [ ] Version compatibility noted
- [ ] Cross-references updated

## Contributing Process

### For New Contributors

1. **Read existing tutorials**: Understand the style
2. **Start small**: Fix typos, update examples
3. **Ask questions**: Open an issue first
4. **Follow templates**: Use existing structure
5. **Test thoroughly**: Run all examples

### Pull Request Template

```markdown
## Tutorial Update: [Brief Description]

### Changes Made
- [ ] Added new content
- [ ] Updated examples
- [ ] Fixed errors
- [ ] Updated links

### Testing Done
- [ ] All INI configs tested
- [ ] Scripts executed successfully
- [ ] Links verified
- [ ] Markdown rendered correctly

### Files Modified
- tutorials/[filename]

### Compatibility
- Works with: nkululeko >= X.X.X
- Tested on: Python X.X, OS

### Screenshots (if applicable)
[Attach any relevant output/screenshots]
```

## Feedback and Improvements

### Collecting Feedback

Monitor:
- GitHub issues mentioning "tutorial" or "activation"
- User questions in discussions
- Documentation bugs reported
- Feature requests

### Metrics to Track

- Tutorial completion rate (if analytics available)
- Common error patterns
- Frequently asked questions
- Example usage statistics

## Contact and Support

### Documentation Maintainers
- Check `CONTRIBUTORS.md` for current maintainers
- Open GitHub issues for questions
- Tag documentation-related issues with `docs` label

### Getting Help

If you need help with documentation:
1. Check existing tutorials for similar examples
2. Read `CONTRIBUTING.md` for general guidelines
3. Open an issue with `question` label
4. Ask in GitHub Discussions

## Resources

### Helpful Links
- [Markdown Guide](https://www.markdownguide.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Technical Writing Guide](https://developers.google.com/tech-writing)

### Internal References
- Main documentation: `ini_file.md`
- Contribution guide: `CONTRIBUTING.md`
- Changelog: `CHANGELOG.md`
- Examples directory: `examples/`

---

**Version**: 1.0  
**Last Updated**: 2026-01-09  
**Maintainers**: See CONTRIBUTORS.md
