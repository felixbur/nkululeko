# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'nkululeko'
copyright = '2023, Felix Burkhardt, Bagus Tris Atmaja'
author = 'Felix Burkhardt, Bagus Tris Atmaja'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    # 'nbsphinx',           # to embedd ipynb files
    'sphinx.ext.mathjax',  # to enable latex
    'sphinx_rtd_theme',
    'myst_parser',        # to enable markdown
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# suffix for source files
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# headed anchors until h3
myst_heading_anchors = 3

# The master toctree document.
master_doc = 'index'


# html_theme_options = {

#     # Set the name of the project to appear in the navigation.
# 'nav_title': 'Nkululeko',

#     # Set the color and the accent color
#     'color_primary': 'blue',
#     'color_accent': 'light-blue',

#     # Set the repo location to get a badge with stats
#     'repo_url': 'https://github.com/felixbur/nkululeko',
#     'repo_name': 'Machine learning speaker characteristics',

#     # Visible levels of the global TOC; -1 means unlimited
#     'globaltoc_depth': 3,
#     # If False, expand all TOC entries
#     'globaltoc_collapse': True,
#     # If True, show hidden TOC entries
#     'globaltoc_includehidden': False,
#     #'logo_icon': "&#xe913;"
# }
