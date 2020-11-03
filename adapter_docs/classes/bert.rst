BERT
======

The BERT model was proposed in `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__
by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It is a bidirectional transformer
pre-trained using a combination of masked language modeling objective and next sentence prediction.

.. note::
    This class is nearly identical to the PyTorch implementation of BERT in Huggingface Transformers.
    For more information, visit `the corresponding section in their documentation <https://huggingface.co/transformers/model_doc/bert.html>`_.

BertConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertConfig
    :members:


BertTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


BertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertModel
    :members:


BertModelWithHeads
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertModelWithHeads
    :members:


BertForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForPreTraining
    :members:


BertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForMaskedLM
    :members:


BertForNextSentencePrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForNextSentencePrediction
    :members:


BertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForSequenceClassification
    :members:


BertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForMultipleChoice
    :members:


BertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForTokenClassification
    :members:


BertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForQuestionAnswering
    :members:
