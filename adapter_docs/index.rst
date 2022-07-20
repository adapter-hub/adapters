.. adapter-transformers documentation master file, created by
   sphinx-quickstart on Sat Apr 18 10:21:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AdapterHub Documentation
================================================

*AdapterHub* is a framework simplifying the integration, training and usage of adapters and other efficient fine-tuning methods for Transformer-based language models.
For a full list of currently implemented methods, see the `table in our repository <https://github.com/adapter-hub/adapter-transformers#implemented-methods>`_.

The framework consists of two main components:

- ``adapter-transformers``, an extension of Huggingface's `Transformers <https://huggingface.co/transformers/>`_ library that adds adapter components to transformer models

- `The Hub <https://adapterhub.ml>`_, a central repository collecting pre-trained adapter modules

Currently, we support the PyTorch versions of all models as listed on the `Model Overview <model_overview.html>`_ page.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   overview
   training

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   adapter_composition
   prediction_heads
   embeddings
   extending
   transitioning

.. toctree::
   :maxdepth: 2
   :caption: Loading and Sharing

   loading
   hub_contributing
   huggingface_hub

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   model_overview
   classes/models/auto
   classes/models/bart
   classes/models/bert
   classes/models/deberta
   classes/models/deberta_v2
   classes/models/distilbert
   classes/models/encoderdecoder
   classes/models/gpt2
   classes/models/mbart
   classes/models/roberta
   classes/models/t5
   classes/models/vit
   classes/models/xlmroberta

.. toctree::
   :maxdepth: 2
   :caption: Adapter-Related Classes

   classes/adapter_config
   classes/model_adapters_config
   classes/adapter_modules
   classes/adapter_layer
   classes/model_mixins
   classes/adapter_utils

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing
   contributing/adding_adapter_methods
   contributing/adding_adapters_to_a_model

Citation
========

.. code-block:: bibtex

   @inproceedings{pfeiffer2020AdapterHub,
      title={AdapterHub: A Framework for Adapting Transformers},
      author={Jonas Pfeiffer and
               Andreas R\"uckl\'{e} and
               Clifton Poth and
               Aishwarya Kamath and
               Ivan Vuli\'{c} and
               Sebastian Ruder and
               Kyunghyun Cho and
               Iryna Gurevych},
      booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020): Systems Demonstrations},
      year={2020},
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/2020.emnlp-demos.7",
      pages = "46--54",
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
