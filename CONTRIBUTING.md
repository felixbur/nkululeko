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
	-> Please never work directly on the `master` branch!
	```

6.	Once you are done, make sure to format the code using black to fit Nkululeko's codestyle.

	```bash
	black nkululeko/
	isort --profile black nkululeko/
	# Alternatively and additionaly, use ruff:
	ruff check --fix --output-format=full nkululeko
	```

7.	Make sure that the tests succeed and have enough coverage.

	```./run_tests2 all ```

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

Note
----

This document was based on the [scikit-learn](http://scikit-learn.org/) & [librosa](https://github.com/librosa/librosa) contribution guides.
