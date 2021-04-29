BART
=====

The Bart model was proposed in `BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
Translation, and Comprehension <https://arxiv.org/abs/1910.13461>`__ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.

According to the abstract,

- Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a
  left-to-right decoder (like GPT).
- The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme,
  where spans of text are replaced with a single mask token.
- BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It
  matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new
  state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains
  of up to 6 ROUGE.

.. note::
    This class is nearly identical to the PyTorch implementation of BART in Huggingface Transformers.
    For more information, visit `the corresponding section in their documentation <https://huggingface.co/transformers/model_doc/bart.html>`_.


BartConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartConfig
    :members:


BartTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartTokenizer
    :members:



BartModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartModel
    :members: forward


BartModelWithHeads
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartModelWithHeads
    :members:
    :inherited-members: BartPretrainedModel


BartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForConditionalGeneration
    :members: forward


BartForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForSequenceClassification
    :members: forward


BartForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForQuestionAnswering
    :members: forward
