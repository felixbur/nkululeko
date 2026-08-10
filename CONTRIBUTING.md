Contributing guidelines
=======================

How to contribute
-----------------

The preferred way to contribute to nkululeko is to fork the [main repository](https://github.com/felixbur/nkululeko) on GitHub:

1.	Fork the [project repository](https://github.com/felixbur/nkululeko): click on the 'Fork' button near the top of the page. This creates a copy of the code under your account on the GitHub server.

2.	Clone this copy to your local disk:

	-	Using SSH:

	```bash
	git clone git@github.com:YourLogin/nkululeko.git
	cd nkululeko
	```

	-	Using HTTPS:

	```bash
	git clone https://github.com/YourLogin/nkululeko.git
	cd nkululeko
	```

3.	Remove any previously installed nkululeko versions, then install your local copy with testing dependencies:

	```bash
	pip uninstall nkululeko
	pip install .
	```

4.	Create a branch to hold your changes:

	```bash
	git checkout -b my-feature
	```

5.	Start making changes.

	```diff
	-> Please never work directly on the `main` branch!
	```

6.	Once you are done, make sure to format the code using black to fit Nkululeko's codestyle.

	```bash
	black nkululeko/
	isort --profile black nkululeko/
	# Alternatively and additionaly, use ruff:
	ruff check --fix --output-format=full nkululeko
	```

7.	Make sure that the tests succeed and have enough coverage.

	```python3 -m pytest```

8.	Use Git for the to do the version controlling of this copy. When you're done editing, you know the drill `add`, `commit` then `push`:

	```bash
	git add modified_files
	git commit
	```

	to record your changes in Git, push them to GitHub with:

	```bash
	git push -u origin my-feature
	```

9.	Finally, go to the web page of your nkululeko fork repo, and click 'Pull request' button to send your changes to the maintainers to review.

Remarks
-------

It is recommended to check that your contribution complies with the following rules before submitting a pull request:

-	All public methods should have informative docstrings with sample usage presented.

	You can also check for common programming errors with the following tools:

-	Check code formatting using black:

	```bash
	black --check nkululeko
	```

Filing bugs
-----------

We use Github issues to track all bugs and feature requests. In the case of coming across a bug, having a question or a feature suggestion etc. please feel free to open an issue. 

Please check that your issue complies with the following rules before submitting:

-	Verify that your issue is not being currently addressed by in other [issues](https://github.com/felixbur/nkululeko/issues) or [pull requests](https://github.com/felixbur/nkululeko/pulls).

-	Please ensure all code snippets and error messages are formatted appropriately. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-	Please include your operating system type and version number, as well as your Python, nkululeko, numpy, pandas, and scipy versions. This information can be found by running the following code snippet:

	```python
	import sys
	import numpy
	import pandas
	import sklearn
	import nkululeko
	import platform

	print(platform.platform())
	print("Python", sys.version)
	print("NumPy", numpy.__version__)
	print("Pandas", pandas.__version__)
	print("Scikit-learn", sklearn.__version__)
	print("nkululeko", nkululeko.__version__)
	```

Internal data conventions
--------------------------

-	`class_label`: whenever a dataset is labeled, the `class_label` column holds a
	backup of the target column's values *before* any integer/label encoding
	(and, for binned continuous targets, the human-readable bin name; for an
	unlabeled test split filled with a placeholder target, the placeholder
	value). For any split (`df_train`/`df_test`/`df_dev`) that is non-empty
	and has the target column, it is guaranteed to be present by the time
	`Datasplitter.fill_train_and_tests()` returns (see
	`Datasplitter._ensure_class_label` in `nkululeko/data/datasplitter.py`),
	which is the single place responsible for creating it if no earlier step
	(e.g. `Dataset.load()`, `Dataset.map_labels()`) already did. Code reading
	`class_label` outside that guarantee (e.g. before `fill_train_and_tests()`
	has run) must still guard with `"class_label" in df.columns`. If you add a
	new code path that produces or mutates one of these split DataFrames
	before encoding, make sure `class_label` is preserved or backfilled -
	never overwrite it unconditionally, since an earlier step may have already
	populated it with a more informative value (e.g. binned class names).

-	Config defaults: every `config_val()`/`config_val_bool()`/`config_val_list()`/
	`config_val_data()` call site's `(section, key, default)` is scanned by
	`scripts/gen_defaults_table.py` into
	[`docs/source/config_defaults_reference.md`](docs/source/config_defaults_reference.md),
	which also flags keys whose default disagrees across call sites. CI runs
	`python scripts/gen_defaults_table.py --check` and fails if that file is
	stale, so after adding/changing a default, run
	`python scripts/gen_defaults_table.py --write` and commit the result.

Path handling / SonarCloud conventions
---------------------------------------

-	Any CLI argument or other externally-supplied value that becomes a
	filesystem path (e.g. an `--outdir`/`--outfile`-style `argparse` option)
	must be resolved through `safe_path()` in `nkululeko/utils/files.py`
	**called directly, immediately before the value is used** at the
	filesystem call (`os.makedirs`, `open`, `to_csv`, etc.) — not wrapped
	inside another function, even a small validation helper that just
	forwards to `safe_path()` and returns its result. SonarCloud's Python
	taint analyzer (`pythonsecurity:S8707` and related path-injection rules)
	only credits `safe_path()` as clearing the taint when it sits with no
	custom function boundary between it and the sink; when a wrapper
	function stands in between, the analyzer keeps flagging the sink as
	unvalidated even though the path genuinely is validated. This was
	confirmed empirically while fixing `nkululeko/avqi.py`: routing
	`--outdir`/`--outfile` through a `_validate_output_path()` helper that
	called `safe_path()` internally left the SonarCloud finding open across
	two rescans; inlining `safe_path()` directly at each sink (matching the
	existing pattern in `bundle.py`/`infer.py`) closed it. Any additional
	checks (parent-must-exist, existing-entry-kind, etc.) should be done via
	side-effecting calls that take the *already-resolved* path and don't
	produce a new value used to reassign the sink variable.

Note
----

This document was based on the [scikit-learn](http://scikit-learn.org/) & [librosa](https://github.com/librosa/librosa) contribution guides.
