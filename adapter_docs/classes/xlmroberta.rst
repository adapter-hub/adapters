XLM-RoBERTa
============

The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__
by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán,
Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's RoBERTa model released in 2019.
It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

.. note::
    This class is nearly identical to the PyTorch implementation of XLM-RoBERTa in Huggingface Transformers.
    For more information, visit `the corresponding section in their documentation <https://huggingface.co/transformers/model_doc/xlmroberta.html>`_.

XLMRobertaConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaConfig
    :members:


XLMRobertaTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


XLMRobertaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaModel
    :members:


XLMRobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForMaskedLM
    :members:


XLMRobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForSequenceClassification
    :members:


XLMRobertaForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForMultipleChoice
    :members:


XLMRobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForTokenClassification
    :members:
