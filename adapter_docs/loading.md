# Loading Pre-Trained Adapters

## Using adapters from Adapter-Hub

Suppose we have loaded a pre-trained transformer model from Huggingface, e.g. BERT:

```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')
```

We can now easily load a pre-trained adapter module from Adapter Hub by its identifier:

```python
model.load_adapter('sst')
```

In the minimal case, that's everything we need to specify to load a pre-trained task adapter for sentiment analysis, trained on the `sst` dataset using BERT base and a suitable adapter configuration.
To examine what's happening underneath in a bit more detail, let's first write out the full method call with all relevant arguments explicitly stated:

```python
model.load_adapter('sst', AdapterType.text_task, config='pfeiffer', model_name='bert-base-uncased', version=1, load_as='sst')
```

We will go through the different arguments and their meaning one by one:

- The first argument passed to the method specifies the name of the adapter we want to load from Adapter-Hub. The library will search for an available adapter module with this name that matches the model architecture as well as the adapter type and configuration we requested. As the identifier `sst` resolves to a unique entry in the Hub, the corresponding adapter can be successfully loaded based on this information. To get an overview of all available adapter identifiers, please refer to [the Adapter-Hub website](https://adapterhub.ml/explore). The different format options of the identifier string are further described in [How adapter resolving works](#how-adapter-resolving-works).

- The second argument specifies the type of adapter we want to load. In this case, we load a *task* adapter which is the default setting if we don't explicitly state this argument. All other possible adapter types are defined in the `AdapterType` (e.g. we could load a language adapter using `AdapterType.text_lang`) enumeration and explained in more detail on the [Adapter Types](/adapter_types) page.

- The `config` argument defines the adapter architecture the loaded adapter should have.
The value of this parameter can be either a string identifier for one of the predefined architectures, the identifier of an architecture available in the Hub or a dictionary representing a full adapter configuration.
Based on this information, the library will only search for pre-trained adapter modules having the same configuration.

```eval_rst
.. tip::
    If the config parameter is not specified, the loading method will fall back to default adapter architectures, first for the requested adapter type and then to a global default. To set the default architecture for an adapter type, you can use ``model.set_adapter_config()``, e.g. ``model.set_adapter_config(AdapterType.text_task, 'houlsby')``. Now, ``load_adapter()`` would always search for adapters in ``houlsby`` architecture by default.
```

- Adapter modules trained on different pre-trained language models in general can not be used interchangeably.
Therefore, we need to make sure to load an adapter matching the language model we are using.
If possible, the library will infer the name of the pre-trained model automatically (e.g. when we use `from_pretrained('identifier')` to load a model from Huggingface). However, if this is not the case, we must specify the name of the host model in the `model_name` parameter.

- There could be multiple versions of the same adapter available. To load a specific version, use the `version` parameter.

- By default, the `load_adapter()` method will add the loaded adapter using the identifier string given as the first argument.
To load the adapter using a custom name, we can use the `load_as` parameter.

## How adapter resolving works

As described in the previous section, the methods for loading adapters are able to resolve the correct adapter weights
based on the given identifier string, the model name and the adapter configuration.
Using this information, the `adapter-transformers` library searches for a matching entry in the index of the [Hub GitHub repo](https://github.com/adapter-hub/hub).

The identifier string used to find a matching adapter follows a format consisting of three components:
```
<task>/<subtask>@<username>
```

- `<task>`: A generic task identifier referring to a category of similar tasked (e.g. `sentiment`, `nli`)
- `<subtask>`: A dataset or domain, on which the adapter was trained (e.g. `multinli`, `wiki`)
- `<username>`: The name of the user or organization that uploaded the pre-trained adapter

An example of a full identifier following this format might look like `qa/squad1.1@example-org`.

```eval_rst
.. important::
    In many cases, you don't have to give the full string identifier with all three components to successfully load an adapter from the Hub. You can drop the `<username>` you don't care about the uploader of the adapter.  Also, if the resulting identifier is still unique, you can drop the ``<task>`` or the ``<subtask>``. So, ``qa/squad1.1``, ``squad1.1`` or ``squad1.1@example-org`` all may be valid identifiers.
```

For more background information on the identifier string format and the Hub index structure, you can also refer to the [specification document](https://github.com/adapter-hub/hub/blob/master/spec.md) on GitHub.
