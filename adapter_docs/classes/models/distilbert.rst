DistilBERT
===========

The DistilBERT model was proposed in the blog post
`Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT <https://medium.com/huggingface/distilbert-8cf3380435b5>`__,
and the paper `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/abs/1910.01108>`__.
DistilBERT is a small, fast, cheap and light Transformer model trained by distilling Bert base. It has 40% less
parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of Bert's performances as measured on
the GLUE language understanding benchmark.

.. note::
    This class is nearly identical to the PyTorch implementation of DistilBERT in Huggingface Transformers.
    For more information, visit `the corresponding section in their documentation <https://huggingface.co/transformers/model_doc/distilbert.html>`_.


DistilBertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertConfig
    :members:


DistilBertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertTokenizer
    :members:


DistilBertTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertTokenizerFast
    :members:


DistilBertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertModel
    :members:


DistilBertModelWithHeads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertModelWithHeads
    :members:


DistilBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForMaskedLM
    :members:


DistilBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForSequenceClassification
    :members:


DistilBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForQuestionAnswering
    :members:
