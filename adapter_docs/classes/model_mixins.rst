Model Mixins
=======================

These classes provide the basis of adapter module integration into model classes such as adapter saving and loading.
Depending on the model, one of these mixins should be implemented by every adapter-supporting model class.

InvertibleAdaptersMixin
----------------------------------

.. autoclass:: transformers.InvertibleAdaptersMixin
    :members:

ModelAdaptersMixin
------------------

.. autoclass:: transformers.ModelAdaptersMixin
    :members:

ModelWithHeadsAdaptersMixin
----------------------------------

.. autoclass:: transformers.ModelWithHeadsAdaptersMixin
    :members:

ModelWithFlexibleHeadsAdaptersMixin
---------------------------------------

.. autoclass:: transformers.ModelWithFlexibleHeadsAdaptersMixin
    :members:

PushAdapterToHubMixin
----------------------

.. autoclass:: transformers.adapters.hub_mixin.PushAdapterToHubMixin
    :members:
