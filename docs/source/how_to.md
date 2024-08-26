# Nkululeko Workflow

Nkululeko is a framework to build machine learning models that recognize
speaker characteristics on a very high level of abstraction (i.e.
starting without programming experience).

This post is meant to help you with setting up your first experiment,
based on the Berlin Emodb.

0) Set up Python

It's written in python, so first you have to set up a Python environment.
It is recommended to use Linux-based systems for easiness, but it should work on Windows as well.
Python is pre-installed on most Linux systems, but you might want to install a virtual environment to keep your system clean.
The current version of nkululeko is tested with Python 3.9.

1) Install Nkululeko

Inside your virtual environment, run:

    pip install nkululeko

This should install nkululeko and all required modules. It takes a long
time and a lot of space, when done initially.

2) Get a database

Load the Berlin emodb database to some location on you harddrive, as
discussed in [this post](http://blog.syntheticspeech.de/2021/08/10/get-all-information-from-emodb/). 
I will refer to the location as "emodb root" from now on. You can also follow [Hello World page](hello_world_aud.md).


3) Adapt the INI file

Use your favourite editor, e.g., Visual Studio code and edit the file
that defines your experiment. You might start with this demo sample. You
can find more templates to start here and an overview on all the options
you can set here

Put the emodb root folder as the emodb value, for me this looks like
this:

    emodb = /home/felix/data/audb/emodb

An overview on all nkululeko options is listed in [INI file](ini_file).

4) Run the experiment

Inside a shell type (or use VSC) and start the process with:

    python -m nkululeko.nkululeko --config exp_emodb.ini

5)  Inspect the results

If all goes well, the program should start by extracting opensmile
features, and, if you\'re done, you should be able to inspect the
results in the folder named like the experiment: exp\_emodb. There
should be a subfolder with a confusion matrix named [images]{.title-ref}
and a subfolder for the textual results named [results]{.title-ref}.
