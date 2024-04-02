# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html
import os
import sys
import re


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default="."), "../src")
sys.path.insert(0, rootdir)


# -- Project information -----------------------------------------------------

project = "AdapterHub"
copyright = "2020-2024, AdapterHub Team"
author = "AdapterHub Team"

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
smv_branch_whitelist = r"^main$"

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None


def skip_head_member(app, what, name, obj, skip, options):
    if type(obj).__name__ == "function" and "inherited-members" in options and (m := re.match(r"add\_(.*)\_head$", name)):
        cls_name = options["inherited-members"].replace("PreTrainedModel", "AdapterModel").replace("PretrainedModel", "AdapterModel")
        cls = vars(sys.modules["adapters"])[cls_name]
        # HACK: currently parses head type from name
        head_type_str = m.group(1).replace("qa", "question_answering")
        if head_type_str in cls.head_types:
            return False
        else:
            return True

    return skip


def setup(app):
    app.connect('autodoc-skip-member', skip_head_member)
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_css_file("custom.css")
