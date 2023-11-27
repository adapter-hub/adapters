# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default="."), "../src")
sys.path.insert(0, rootdir)


# -- Project information -----------------------------------------------------

project = "adapter-transformers"
copyright = "2020-2022, Adapter-Hub Team"
author = "Adapter-Hub Team"

docs_versions = [
    "adapters1.1.1",
    "adapters2.3.0",
    "adapters3.2.1",
]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "sphinx_markdown_tables",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# MyST parser markdown options
myst_heading_anchors = 3
myst_enable_extensions = [
    "dollarmath",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "logo.png"
html_favicon = "favicon.png"


# -- Options for sphinx-multiversion ------------------------------------------

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r"({})".format("|".join([v.replace(".", r"\.") for v in docs_versions]))

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r"^master$"

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None


def setup(app):
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_css_file("custom.css")
