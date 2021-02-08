.. adapter-transformers documentation master file, created by
   sphinx-quickstart on Sat Apr 18 10:21:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AdapterHub Documentation
================================================

*AdapterHub* is a framework simplifying the integration, training and usage of adapter modules for Transformer-based language models.
It integrates adapters for downstream tasks (`Houlsby et al., 2019 <https://arxiv.org/pdf/1902.00751>`_), adapters for cross-lingual transfer (`Pfeiffer et al., 2020a <https://arxiv.org/pdf/2005.00052>`_) and *AdapterFusion* (`Pfeiffer et al., 2020b <https://arxiv.org/pdf/2005.00247>`_).

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
   adapters
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

   classes/adapter_config
   classes/model_adapters_config
   classes/adapter_modules
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
   classes/distilbert


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
