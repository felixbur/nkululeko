# Nkululeko Documentation  

## Documentation URL: [https://nkululeko.readthedocs.io/latest/](https://nkululeko.readthedocs.io/latest/)

## How to run locally

```bash
# Install requirements (from docs directory)
$ uv pip install -r requirements.txt
$ make html
```

It assumed that the current working directory is `docs/` instead of Nkululeko
parent directory. Otherwise, it will install requirements for Nkululeko development,
not for building the documentation.

After that, check the built HTML in `build/html/index.html`

```bash
firefox build/html/index.html
```

## Generate RST files from docstring

To generate RST files from docstring, you can use the following command:

```bash
sphinx-apidoc -f -o source/ ../nkululeko/
```

If found any issues, please report them [here](https://github.com/felixbur/nkululeko/issues).

## Tutorials

### Activation Functions in Neural Networks
- **Main Tutorial**: [tut_activation_functions.md](tut_activation_functions.md)
- **Quick Reference**: [activation_functions_quickref.md](activation_functions_quickref.md)
- **Working Examples**: See `../tutorials/` directory

Learn how to use different activation functions (ReLU, Leaky ReLU, Tanh, Sigmoid) in MLP models.

### Other Documentation
- **Class Diagram**: [class_diagram.md](class_diagram.md)
- **Balancing**: [clustercentroids_balancing.md](clustercentroids_balancing.md)
