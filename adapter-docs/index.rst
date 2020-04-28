.. adapter-transformers documentation master file, created by
   sphinx-quickstart on Sat Apr 18 10:21:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

adapter-transformers Documentation
================================================

This library is an extension of Huggingface's `transformers <https://huggingface.co/transformers/>`_ library that adds Adapter and AdapterFusion components to
transformer models.

The main additions on top of *transformers* are:

- Extension of transformer models by Adapter and AdapterFusion components
- Loading and saving of pre-trained adapters
- *Adapter Hub*, a repository of shared pre-trained adapters


.. toctree::
   :maxdepth: 2
   :caption: General

   installation
   quickstart
   adapter_types
   adapter_training

.. toctree::
   :maxdepth: 2
   :caption: Adapter Hub

   adapter_loading
   adapter_sharing

.. toctree::
   :maxdepth: 2
   :caption: Adapter-Related Classes

   classes/adapter_module
   classes/model_adapters_mixin
   classes/adapters_bert

.. toctree::
   :maxdepth: 1
   :caption: Transformer Models

   classes/bert
   classes/roberta
   classes/xlmroberta


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
