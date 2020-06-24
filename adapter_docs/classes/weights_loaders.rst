Weights Loaders
=======================

These classes perform the extraction, saving and loading of module weights to and from the file system.
All type-specific loader classes inherit from the common ``WeightsLoader`` base class which can also be extended
to add support for additional custom modules.

These classes provide the basis of adapter module integration into model classes such as adapter saving and loading.
Depending on the model, one of these mixins should be implemented by every adapter-supporting model class.

WeightsLoader
------------------

.. autoclass:: transformers.WeightsLoader
    :members:

AdapterLoader
---------------------------

.. autoclass:: transformers.AdapterLoader
    :members:

PredictionHeadLoader
---------------------------

.. autoclass:: transformers.PredictionHeadLoader
    :members:

WeightsLoaderHelper
-------------------

.. autoclass:: transformers.WeightsLoaderHelper
    :members:
