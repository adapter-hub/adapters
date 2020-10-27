RoBERTa
========

The RoBERTa model was proposed in `RoBERTa: A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_
by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
Veselin Stoyanov. It is based on Google's BERT model released in 2018.

.. note::
    This class is nearly identical to the PyTorch implementation of RoBERTa in Huggingface Transformers.
    For more information, visit `the corresponding section in their documentation <https://huggingface.co/transformers/model_doc/roberta.html>`_.

RobertaConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaConfig
    :members:


RobertaTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


RobertaModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaModel
    :members:


RobertaModelWithHeads
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaModelWithHeads
    :members:


RobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForMaskedLM
    :members:


RobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForSequenceClassification
    :members:


RobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForTokenClassification
    :members:
