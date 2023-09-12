Whisper
-----------------------------------------------------------------------------------------------------------------------

The Whisper model was presented in `Robust Speech Recognition via Large-Scale Weak Supervision
<https://arxiv.org/abs/2212.04356>`_ by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine
McLeavey, Ilya Sutskever.

According to the abstract, Whisper is trained on 680,000 hours of multilingual and multitask data. This
scale was previously unseen. Whisper is able to approach the accuracy and robustness of humans.


WhisperAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.adapters.WhisperAdapterModel
    :members:
    :inherited-members: WhisperPreTrainedModel
