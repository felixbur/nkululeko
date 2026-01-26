# Nkululeko Documentation  

## Documentation URL: [https://nkululeko.readthedocs.io/latest/](https://nkululeko.readthedocs.io/latest/)

## Recent Updates

- **EER Metric**: Added Equal Error Rate (EER) metric for binary classification tasks (deepfake detection, speaker verification). See [EER_IMPLEMENTATION.md](source/EER_IMPLEMENTATION.md) for details.

## How to run locally

```bash
# Install requirements (from docs directory)
$ uv pip install -r requirements.txt
$ make html
```

It assumed that the current working directory is `docs/` instead of Nkululeko
parent directory. Otherwise, it will install requirements for Nkululeko development, not for building the documentation.

After that, check the built HTML in `build/html/index.html`

```bash
firefox build/html/index.html
```

Documentation is written using [Sphinx](https://www.sphinx-doc.org/en/master/)  in markdown format with [MyST parser](https://myst-parser.readthedocs.io/en/latest/). The main source files are in `docs/source/` directory.  

## Generate RST files from docstring

To generate RST files from docstring, you can use the following command:

```bash
sphinx-apidoc -f -o source/ ../nkululeko/
```

If found any issues, please report them [here](https://github.com/felixbur/nkululeko/issues).

