.. adapter-transformers documentation master file, created by
   sphinx-quickstart on Sat Apr 18 10:21:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AdapterHub Documentation
================================================

*AdapterHub* is a framework simplifying the integration, training and usage of adapter modules for Transformer-based language models.

The framework consists of two main components:

- ``adapter-transformers``, an extension of Huggingface's `Transformers <https://huggingface.co/transformers/>`_ library that adds adapter components to transformer models

- `The Hub <https://adapterhub.ml>`_, a central repository collecting pre-trained adapter modules

The *adapter-transformers* section documents the integration of adapters into the ``transformers`` library and how training adapters works.

The section on *Adapter-Hub* describes the fundamentals of the pre-trained adapter repository and how to contribute new adapters.

Currently, we support the PyTorch versions of all models listed in the *Supported Models* section.

.. toctree::
   :maxdepth: 2
   :caption: adapter-transformers

   installation
   quickstart
   adapter_types
   training
   prediction_heads
   extending

.. toctree::
   :maxdepth: 2
   :caption: Adapter-Hub

   loading
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Adapter-Related Classes

   classes/adapter_modules
   classes/adapter_config
   classes/model_mixins
   classes/bert_mixins
   classes/adapter_utils
   classes/weights_loaders

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
