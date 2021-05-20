# Introduction to Adapters

Adapters have been introduced as a new alternative of fine-tuning language models on a downstream task.
Instead of fine-tuning the full model, a small set of newly introduced task-specific parameters is updated during fine-tuning.
The rest of the model is kept fix.
Adapters provide advantages in terms of size, modularity and composability while often achieving results on-par with full fine-tuning.
We will not go into detail about the theoretical background of adapters in the following but refer to some literature providing more explanations here:

* [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf) (Houlsby et al., 2019)
* [Simple, Scalable Adaptation for Neural Machine Translation](https://arxiv.org/pdf/1909.08478.pdf) (Bapna and Firat, 2019)
* [MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf) (Pfeiffer et al., 2020)
* [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/pdf/2007.07779.pdf) (Pfeiffer et al., 2020)

```eval_rst
.. note::
    *AdapterHub* aims to support a variety adapter setups. To understand our implementation of adapters, there's some important terminology to grasp at first: the difference between *adapter types* and *adapter architectures*. 
    
    The *adapter type* describes for which purpose an adapter is used whereas the *adapter architecture* (or *adapter configuration*) describes from which components the adapter modules in the language model are constructed.
```

## Adapter types

The adapter type defines the purpose of an adapter. Currently, all adapters are categorized as one of the following types:

- **Task adapter**: Task adapters are fine-tuned to learn representations for a specific downstream tasks such as sentiment analysis, question answering etc. Task adapters for NLP were first introduced by [Houlsby et al., 2019](https://arxiv.org/pdf/1902.00751.pdf).

- **Language adapter**: Language adapters are used to learn language-specific transformations. After being trained on a language modeling task, a language adapter can be stacked before a task adapter for training on a downstream task. To perform zero-shot cross-lingual transfer, one language adapter can simply be replaced by another. In terms of architecture, language adapters are largely similar to task adapters, except for an additional _invertible adapter_ layer after the embedding layer. This setup was introduced and is further explained by [Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00052.pdf).

Beginning with version 2, both adapter types are treated identically within the library.
The additional invertible adapters are defined via the adapter configuration (see next section).
In v1.x, the distinction between task and language adapters was made with the help of the `AdapterType` enumeration.

## Adapter architectures

```eval_rst
.. figure:: img/architecture.png
    :width: 350
    :align: center
    :alt: Adapter architectures

    Visualization of possible adapter configurations with corresponding dictionary keys.
```

The concrete structure of adapter modules and their location in the layers of a Transformer model is specified by a configuration dictionary.
This is referred to as the adapter architecture.
The currently possible configuration options are visualized in the figure above.
When adding new adapters using the `add_adapter()` method, the configuration can be set providing the `config` argument.
The passed value can be either a plain Python dict containing all keys pictured or a subclass of [`AdapterConfig`](classes/adapter_config.html#transformers.AdapterConfig).

For convenience, `adapter-transformers` has some common architectures built-in:
- [`HoulsbyConfig`](classes/adapter_config.html#transformers.HoulsbyConfig) as proposed by [Houlsby et al., 2019](https://arxiv.org/pdf/1902.00751.pdf)
- [`PfeifferConfig`](classes/adapter_config.html#transformers.PfeifferConfig) as proposed by [Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00052.pdf)

Both of these classes have counterparts with invertible adapters, typically used as language adapters:
[`HoulsbyInvConfig`](classes/adapter_config.html#transformers.HoulsbyInvConfig) and [`PfeifferInvConfig`](classes/adapter_config.html#transformers.PfeifferInvConfig).

Furthermore, pre-defined architectures can be loaded from the Hub:

```python
# load "pfeiffer" config from Hub, but replace the reduction factor
config = AdapterConfig.load("pfeiffer", reduction_factor=12)
# add a new adapter with the loaded config
model.add_adapter("dummy", config=config)
```

 You can also [add your own architecture to the Hub](contributing.html#add-a-new-adapter-architecture).
