# Nkululeko documentation  

## Documentation URL: [https://nkululeko.readthedocs.io/en/latest/](https://nkululeko.readthedocs.io/en/latest/)

## How to run locally

```bash
# Install requirements (from docs directory)
$ pip install -r requirements.txt
$ make html
```

It assumed that the current working directory is `docs/` instead of Nkululeko
parent directory. Otherwise, it will install requirements for Nkululeko development,
not for building the documentation.

After that, check the built HTML in `build/html/index.html`

```bash
firefox build/html/index.html
```

If found any issues, please report them [here](https://github.com/felixbur/nkululeko/issues).
