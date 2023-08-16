Installation
----------------

The easiest and safest way to try Nkululeko is using virtualenv, either using venv or mini (conda).  Here an example using Conda. ::


    $ conda create -n nkululeko python=3.8  
    $ conda activate nkululeko  
    $ pip install nkululeko  

And for venv: ::

    $ python3 -m venv nkululeko  
    $ source nkululeko/bin/activate  
    $ pip install nkululeko

For development version, you install directly from the source: ::

    # clone the repository  
    $ git clone https://github.com/felixbur/nkululeko.git  
    $ cd nkululeko  
    # install in editable mode  
    $ pip install -e .  
    # or add the path to your PYTHONPATH
    $ export PYTHONPATH=$PYTHONPATH:/path/to/nkululeko

Check the installation: ::

    # via pip
    pip list | grep nkululeko
    # inside python
    >>> import nkululeko
    >>> nkululeko.__version__
    # to differentiate between the installed and the development version
    >>> nkululeko.__file__

If you see the version of Nkululeko from one of the first two command above, e.g. ``0.55.0``, you are ready to go.