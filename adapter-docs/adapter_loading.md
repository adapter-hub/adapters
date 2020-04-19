# Load Pre-Trained Adapters

## Using adapters from Adapter Hub

Suppose we have loaded a pre-trained transformer model from Huggingface, e.g. BERT:

```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')
```

We can now easily load a pre-trained adapter module from Adapter Hub by its identifier (note that all arguments except for the adapter name are optional):

```python
model.load_adapter('sst', default_config='pfeiffer', version=1, load_head=True)
```

We will go through what is happening here in a bit more detail:

- By calling the `load_adapter()` method we specify that we want to have a *task* adapter. For other types of adapters, similar
methods with other names are available (e.g. `load_language_adapter()` for *language* adapters; see [Adapter Types](/adapter_types) for more).

- The first argument passed to the method is the name of the adapter we want to load. As we load a task adapter in this case,
this refers to the name of the task. Using this identifier, the method will automatically resolve the right adapter weights
matching our model architecture and our adapter configuration. A list of all available adapter names is accessible via
`PRETRAINED_TASK_ADAPTER_MAP` or the corresponding equivalents for other adapter types.

- If our model already contains one or more adapters, `load_adapter()` will search for the adapter weights matching the
adapter configuration of the existent adapters. If no adapter has been added so far, we will search for adapter weights
matching the configuration defined by the `default_config` parameter.

```eval_rst
.. tip::
    Instead of using the ``default_config`` parameter of the load method, you can also set the adapter configuration using
    ``model.set_adapter_config('pfeiffer')`` (or any of its equivalents for other adapter types) before adding the first adapter.
```

- There could be multiple versions of the same adapter available. To load a specific version, use the `version` parameter.

- For many adapters, a corresponding pre-trained prediction head is available. If available, you can load this head together
with the adapter weights by setting `load_head` to `True`.

## How adapter resolving works

As described in the previous section, the methods for loading adapters are able to resolve the correct adapter weights
based on the model architecture and the adapter configuration.

First, a hash containing information on the model architecture and the adapter configuration dictionary is calculated.
Using this hash, the loading mechanism searches for a matching subdirectory in the remote directory defined in the local
mapping between adapter names and download locations. This subdirectory is expected to contain the pretrained adapter weights
and configuration.

You can create a folder structure in the described format by leveraging the `extract_adapters.py` script provided with the repository.
This script will save all adapters of a pre-trained model into this structure.
