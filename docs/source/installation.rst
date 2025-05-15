Installation
----------------

Nkululeko requires Python 3.9 or higher. The easiest and safest way to install Nkululeko is using a virtual environment, either with venv or conda.

Using Conda
^^^^^^^^^^^

.. code-block:: bash

    $ conda create -n nkululeko python=3.9
    $ conda activate nkululeko
    $ pip install nkululeko

Using venv
^^^^^^^^^^

.. code-block:: bash

    $ python3 -m venv nkululeko
    $ source nkululeko/bin/activate  # On Windows: nkululeko\Scripts\activate
    $ pip install nkululeko

Current version: ``0.94.1``

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^

Nkululeko supports optional dependencies through extras. You can install them using the following syntax:

.. code-block:: bash

    # Install with PyTorch support (default PyTorch)
    $ pip install nkululeko[torch]

    # Install with CPU-only PyTorch
    $ pip install nkululeko[torch-cpu]
    # Or manually:
    $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install with Spotlight support
    $ pip install PyYAML  # Install PyYAML first to avoid dependency issues
    $ pip install nkululeko[spotlight]

    # Install with TensorFlow support
    $ pip install nkululeko[tensorflow]

    # Install all optional dependencies
    $ pip install nkululeko[all]

For specific model support:

.. code-block:: bash

    # For SQUIM model (requires nightly PyTorch)
    $ pip install nkululeko[torch-nightly]
    # Or manually:
    $ pip uninstall -y torch torchvision torchaudio
    $ pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^

For the development version, install directly from the source:

.. code-block:: bash

    # Clone the repository
    $ git clone https://github.com/felixbur/nkululeko.git
    $ cd nkululeko
    # Install in editable mode with all dependencies
    $ pip install -e ".[all]"
    # Or with specific extras
    $ pip install -e ".[torch,spotlight]"

Verify Installation
^^^^^^^^^^^^^^^^^^

Check that Nkululeko is installed correctly:

.. code-block:: bash

    # Via pip
    $ pip list | grep nkululeko

    # Inside Python
    >>> import nkululeko
    >>> nkululeko.__version__
    >>> nkululeko.__file__  # Shows installation path

If you see the version of Nkululeko (e.g., ``0.94.1``), you are ready to go.
