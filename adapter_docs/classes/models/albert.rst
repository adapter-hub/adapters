ALBERT
======

The ALBERT model was proposed in `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations <https://arxiv.org/abs/1909.11942>`__
by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
It presents two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

- Splitting the embedding matrix into two smaller matrices.
- Using repeating layers split among groups.

AlbertAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.adapters.AlbertAdapterModel
    :members:
    :inherited-members: AlbertPreTrainedModel
