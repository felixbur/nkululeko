.. nkululeko documentation master file, created by
   sphinx-quickstart on Mon Jul 24 16:18:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nkululeko's documentation!
=====================================
This documentation contains installation, usage, INI file format, and tutorials of Nkululeko, Machine Learning Speaker Characteristics. The program is intended for novice people interested in speaker characteristics detection (e.g., emotion, age, and gender) without proficient in (Python) programming language.  Main features of Nkululeko are:

1. Finding good combinations of several variables, e.g., acoustic features and models (classifier or regressor), feature standardization, augmentation, etc.,  for speaker characteristics detection,
2. Characteristics of the database, such as distribution of gender, age, emotion, duration, data size, and so on with their visualization,
3. Inference of speaker characteristics from a given audio file or streaming audio (can be said also as "weak" labeling for semi-supervised learning).
   
Altogether, this make Nkululeko as a good teaching/learning tool for speaker characteristics detection by machine learning.

The examples only covers some important features of Nkululeko. For more details, please refer to the `Nkululeko Github page <https://felixbur.github.io/nkululeko/>`__ and `Felix's web page <http://blog.syntheticspeech.de/category/nkululeko/>`__. 

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   overview.md
   installation

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage.md
   
.. toctree::
   :maxdepth: 1
   :caption: Architecture

   architecture.md
   glossary.md

.. toctree::
   :maxdepth: 1
   :caption: INI File 

   ini_file.md

.. toctree::
   :maxdepth: 1
   :caption: Visualization

   visualization.md

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   how_to
   hello_world_aud.md
   hello_world_csv.md
   emotion_prediction.md
   comparing
   scaler.md
   optim.md
   test.md
   demo.md
   multidb.md
   ensemble.md
   nkuluflag.md

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   nkululeko.rst
   nkululeko.augmenting.rst
   nkululeko.autopredict.rst
   nkululeko.data.rst
   nkululeko.feat_extract.rst
   nkululeko.losses.rst
   nkululeko.models.rst
   nkululeko.reporting.rst
   nkululeko.segmenting.rst
   nkululeko.utils.rst


.. automodule:: nkululeko
   :members:
   :undoc-members:
   :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
