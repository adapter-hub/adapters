# Installation

Our *adapter-transformers* package is a drop-in replacement for Huggingface's *transformers* library.
It currently supports Python 3.8+ and PyTorch 1.12.1+. You will have to [install PyTorch](https://pytorch.org/get-started/locally/) first. 

```{eval-rst}
.. important::
    ``adapter-transformers`` is a direct fork of ``transformers``.
    This means our package includes all the awesome features of HuggingFace's original package, plus the adapter implementation.
    As both packages share the same namespace, they ideally should not be installed in the same environment.
```

## Using pip

### From PyPI

The simplest way of installation is by using pip to install the package from the Python Package Index:

```
pip install adapter-transformers
```

### From GitHub

You can also install the latest development version directly from our GitHub repository:

```
pip install git+https://github.com/adapter-hub/adapter-transformers.git
```

## From repository

Alternatively, you can clone the repository first and install the package from source.
This allows you to run the included example scripts directly:

```
git clone https://github.com/adapter-hub/adapter-transformers.git
cd adapter-transformers
pip install .
```
