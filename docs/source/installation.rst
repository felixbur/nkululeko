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

Some functionalities require additional packages:

.. code-block:: bash

    # For SQUIM model
    $ pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

    # For spotlight adapter
    $ pip install renumics-spotlight sliceguard

    # For CPU-only PyTorch installation
    $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^

For the development version, install directly from the source:

.. code-block:: bash

    # Clone the repository
    $ git clone https://github.com/felixbur/nkululeko.git
    $ cd nkululeko
    # Install in editable mode
    $ pip install -e .

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
