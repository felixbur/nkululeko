# Testing a Saved Model

> The former `nkululeko.testing` module has been merged into
> `nkululeko.predict`. The same workflow is available via
> `python -m nkululeko.predict --type model ...`. See [predict.md](predict.md)
> for the full reference and [test_module.md](test_module.md) for a tutorial.

## Quick example

Test a previously trained model on a labeled list of files:

```bash
python -m nkululeko.predict \
    --config myconfig.ini \
    --type model \
    --list my_test.csv \
    --outfile myresults.csv
```

This loads the best model from the experiment specified in `myconfig.ini`
(which must have been trained with `MODEL.save = True`) and writes the
predictions next to the original columns in `myresults.csv`.
