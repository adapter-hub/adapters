Adapter Configuration
=======================

Classes representing the architectures of adapter modules and fusion layers.

Single (bottleneck) adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdapterConfigBase
    :members:

.. autoclass:: transformers.AdapterConfig
    :members:
    :inherited-members: Mapping

.. autoclass:: transformers.PfeifferConfig
    :members:

.. autoclass:: transformers.PfeifferInvConfig
    :members:

.. autoclass:: transformers.HoulsbyConfig
    :members:

.. autoclass:: transformers.HoulsbyInvConfig
    :members:

.. autoclass:: transformers.ParallelConfig
    :members:

.. autoclass:: transformers.CompacterConfig
    :members:

.. autoclass:: transformers.CompacterPlusPlusConfig
    :members:

Prefix Tuning
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PrefixTuningConfig
    :members:
    :inherited-members: Mapping

LoRAConfig
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LoRAConfig
    :members:
    :inherited-members: Mapping

IA3Config
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.IA3Config
    :members:
    :inherited-members: Mapping

Combined configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ConfigUnion
    :members:
    :inherited-members: Mapping

.. autoclass:: transformers.MAMConfig
    :members:

.. autoclass:: transformers.UniPELTConfig
    :members:

Adapter Fusion
~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdapterFusionConfig
    :members:
    :inherited-members: Mapping

.. autoclass:: transformers.StaticAdapterFusionConfig
    :members:

.. autoclass:: transformers.DynamicAdapterFusionConfig
    :members:

Adapter Setup
~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdapterSetup
    :members:
