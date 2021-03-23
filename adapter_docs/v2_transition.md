# Transitioning from v1 to v2

Version 2 of `adapter-transformers` brings a lot of new features and changes to the library.
This document gives an overview on what's new and which library changes to look out for when upgrading from v1 to v2.
Potentially breaking changes are marked with "⚠️".

## What's new

### Adapter composition blocks

The new version introduces a radically different way to define adapter setups in a Transformers model,
allowing much more advanced and flexible adapter composition possibilities.
An example setup using this new, modular composition mechanism might look like this:

```python
import transformers.adapters.composition as ac

model.active_adapters = ac.Stack("a", ac.Split("b", "c", split_index=60))
```

As we see, the basic building blocks of this setup are simple objects representing different possibilities to combine single adapters.
In the example, `Stack` describes stacking layers of adapters on top of each other,
as used in the _MAD-X_ framework for cross-lingual transfer.
`Split` describes splitting the input sequences between two adapters at a specified index.
Thus, in the shown setup, in each adapter layer, the input is first passed through adapter `a` before being split up between adapters `b` and `c` and passed through both adapters in parallel.

Besides the two blocks shown, `adapter-transformers` currently also includes a `Fuse` block (for _AdapterFusion_) and a `Parallel` block (see below).
All of these blocks derive from `AdapterCompositionBlock`, and they can be combined in flexibly in many ways.
For more information on specifying the active adapters using `active_adapters` and the new composition blocks,
refer to the [corresponding section in our documentation](adapter_composition.md).

### New model support: Adapters for BART and GPT-2

The two new model architectures added in v2.0.0, BART and GPT-2, start the process of integrating adapters into sequence-to-sequence models, with more to come.

We have [a separate blog post]() presenting our results when training adapters on both models and new adapters in the Hub.

### AdapterDrop

Version 2 of `adapter-transformers` integrates some new ideas introduced in the _AdapterDrop_ paper [(Rückle et al., 2020)](https://arxiv.org/pdf/2010.11918.pdf). This includes _robust_ adapter training by dynamically dropping adapters from random layers in each training step.
Robust _AdapterDrop_ training is presented on an example [in this Colab notebook](https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/Adapter_Drop_Training.ipynb).

Additionally, `adapter-transformers` enables parallel multi-task inference on different adapters via the `Parallel` adapter composition block.
You can find out more about this feature [here](adapter_composition.html#parallel).

### Transformers upgrade

Version 2.0.0 upgrades the underlying HuggingFace Transformers library from v3.5.1 to v4.2.2, bringing many awesome new features created by HuggingFace.

## What has changed

### Unified handling of all adapter types ⚠️

The new version removes the hard distinction between _task_ and _language_ adapters (realized using the `AdapterType` enumeration in v1) everywhere in the library.
Instead, all adapters use the same set of methods.
This, of course, leads to some breaking changes.
For example, you don't have to specify the adapter type anymore when adding a new adapter.
Instead of...
```python
# OLD (v1)
model.add_adapter("name", AdapterType.text_task, config="houlsby")
```
... you would simply write...
```python
# NEW (v2)
model.add_adapter("name", config="houlsby")
```

A similar change applies for loading adapters from the Hub using `load_adapter()`.

In v1, adapters of type `text_lang` automatically had invertible adapter modules added.
As this type distinction is now removed, adding invertible adapters can be specified via the adapter config.
For example...

```python
# OLD (v1)
model.add_adapter("name", AdapterType.text_task, config="pfeiffer")
```
... in v1 would be equivalent to the following in v2:
```python
# NEW (v2)
model.add_adapter("name", config="pfeiffer+inv")
```

### Removal of `adapter_names` parameter in model forward() ⚠️

In v1, it was possible to specify the active adapters using the `adapter_names` parameter in each call to the model's `forward()` method.
With the integration of the new, unified mechanism for specifying adapter setups using composition blocks, this parameter was dropped.
The active adapters now are exclusively set via `set_active_adapters()` or the `active_adapters` property.
For example...

```python
# OLD (v1)
model(**input_data, adapter_names="awesome_adapter")
```
... would become...
```python
# NEW (v2)
model.active_adapters = "awesome_adapter"
model(**input_data)
```

## More (internal) changes

### Changes to adapter weights dictionaries and config ⚠️

With the unification of different adapter types and other internal refactorings, the names of the modules holding the adapters have changed.
This affects the weights dictionaries exported by `save_adapter()`, making the adapters incompatible _in name_.
Nonetheless, rhis does not visibly affect loading older adapters with the new version.
When loading an adapter trained with v1 in a newer version, `adapter-transformers` will automatically convert the weights to the new format.
However, loading adapters trained with newer versions into an earlier v1.x version of the library does not work.

Additionally, there have been some changes in the saved configuration dictionary, also including automatic conversions from older versions.

### Refactorings in adapter implementations

There have been some refactorings mainly in the adapter mixin implementations.
Further details can be found [in the guide for adding adapters to a new model](https://github.com/Adapter-Hub/adapter-transformers/blob/master/adding_adapters_to_a_model.md).
