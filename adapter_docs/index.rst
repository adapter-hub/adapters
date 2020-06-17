.. adapter-transformers documentation master file, created by
   sphinx-quickstart on Sat Apr 18 10:21:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

adapter-transformers Documentation
================================================

``adapter-transformers`` is an extension of Huggingface's `transformers <https://huggingface.co/transformers/>`_ library
that adds adapter components to transformer models.

The main additions on top of *transformers* are:

- Extension of transformer models by task and language adapter components
- Loading and saving of pre-trained adapters
- `Adapter Hub <https://adapterhub.ml>`_, a repository of shared pre-trained adapters

Currently, we support the PyTorch versions of all models listed in the *Supported Models* section.

.. toctree::
   :maxdepth: 2
   :caption: General

   installation
   quickstart
   adapter_types
   training

.. toctree::
   :maxdepth: 2
   :caption: Adapter Hub

   loading
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Adapter-Related Classes

   classes/adapter_modules
   classes/adapter_config
   classes/adapter_model_mixin
   classes/adapter_bert
   classes/adapter_utils

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   classes/bert
   classes/roberta
   classes/xlmroberta


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
