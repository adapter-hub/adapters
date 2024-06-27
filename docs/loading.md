# Loading Pre-Trained Adapters

## Finding pre-trained adapters

**[AdapterHub.ml](https://adapterhub.ml/explore)** provides a central collection of all pre-trained adapters uploaded via Hugging Face's [Model Hub](https://huggingface.co/models).
You can easily find pre-trained adapters for your task of interest along with all relevant information and code snippets to get started.

```{eval-rst}
.. note::
    The original `Hub repository <https://github.com/adapter-hub/hub>`_ (via ``source="ah"``) has been archived and migrated to the HuggingFace Model Hub. The Adapters library supports automatic redirecting to the HF Model Hub when attempting to load adapters from the original Hub repository.
```

Alternatively, [`list_adapters()`](adapters.utils.list_adapters) provides a programmatical way of accessing all available pre-trained adapters.
This will return an [`AdapterInfo`](adapters.utils.AdapterInfo) object for each retrieved adapter.
E.g., we can use it to retrieve information for all adapters trained for a specific model:

```python
from adapters import list_adapters

# source can be "ah" (archived Hub repo), "hf" (huggingface.co) or None (for both, default)
adapter_infos = list_adapters(source="hf", model_name="bert-base-uncased")

for adapter_info in adapter_infos:
    print("Id:", adapter_info.adapter_id)
    print("Model name:", adapter_info.model_name)
    print("Uploaded by:", adapter_info.username)
```

In case the adapter ID is known, information for a single adapter can also be retrieved via [`get_adapter_info()`](adapters.utils.get_adapter_info):

```python
adapter_info = get_adapter_info("@ukp/bert-base-uncased_sentiment_sst-2_pfeiffer", source="ah")

print("Id:", adapter_info.adapter_id)
print("Model name:", adapter_info.model_name)
print("Uploaded by:", adapter_info.username)
```

## Using pre-trained adapters in your code

Suppose we have loaded a pre-trained transformer model from Hugging Face, e.g. BERT, and initialized it for adding adapters:

```python
from transformers import BertModel
import adapters

model = BertModel.from_pretrained('bert-base-uncased')
adaptrers.init(model)
```

We can now easily load a pre-trained adapter module from Adapter Hub by its identifier using the [`load_adapter()`](adapters.ModelWithHeadsAdaptersMixin.load_adapter) method:

```python
adapter_name = model.load_adapter('sst-2')
```

In the minimal case, that's everything we need to specify to load a pre-trained task adapter for sentiment analysis, trained on the `sst-2` dataset using BERT base and a suitable adapter configuration.
The name of the adapter is returned by [`load_adapter()`](adapters.ModelWithHeadsAdaptersMixin.load_adapter), so we can [activate it](adapter_composition.md) in the next step:
```python
model.set_active_adapters(adapter_name)
```

As the second example, let's have a look at how to load an adapter based on the [`AdapterInfo`](adapters.utils.AdapterInfo) returned by the [`list_adapters()`](adapters.utils.list_adapters) method from [above](#finding-pre-trained-adapters):
```python
from adapters import AutoAdapterModel, list_available_adapters

adapter_infos = list_available_adapters(source="ah")
# Take the first adapter info as an example
adapter_info = adapter_infos[0]

model = AutoAdapterModel.from_pretrained(adapter_info.model_name)
model.load_adapter(adapter_info.adapter_id, source=adapter_info.source)
```

### Advanced usage of `load_adapter()`

To examine what's happening underneath in a bit more detail, let's first write out the full method call with all relevant arguments explicitly stated:

```python
model.load_adapter(
    'sst-2',
    config='pfeiffer',
    model_name='bert-base-uncased',
    version=1,
    load_as='sst',
    source='ah'
)
```

We will go through the different arguments and their meaning one by one:

- The first argument passed to the method specifies the name of the adapter we want to load from Adapter-Hub. The library will search for an available adapter module with this name that matches the model architecture as well as the adapter type and configuration we requested. As the identifier `sst-2` resolves to a unique entry in the Hub, the corresponding adapter can be successfully loaded based on this information. To get an overview of all available adapter identifiers, please refer to [the Adapter-Hub website](https://adapterhub.ml/explore).

- The `config` argument defines the adapter architecture the loaded adapter should have.
The value of this parameter can be either a string identifier for one of the predefined architectures, the identifier of an architecture available in the Hub or a dictionary representing a full adapter configuration.
Based on this information, the library will only search for pre-trained adapter modules having the same configuration.

- Adapter modules trained on different pre-trained language models in general can not be used interchangeably.
Therefore, we need to make sure to load an adapter matching the language model we are using.
If possible, the library will infer the name of the pre-trained model automatically (e.g. when we use `from_pretrained('identifier')` to load a model from Hugging Face). However, if this is not the case, we must specify the name of the host model in the `model_name` parameter.

- There could be multiple versions of the same adapter available. To load a specific version, use the `version` parameter.

- By default, the `load_adapter()` method will add the loaded adapter using the identifier string given as the first argument.
To load the adapter using a custom name, we can use the `load_as` parameter.

- Finally the `source` parameter provides the possibility to load adapters from alternative adapter repositories.
Besides the default value `ah`, referring to AdapterHub, it's also possible to pass `hf` to [load adapters from Hugging Face's Model Hub](huggingface_hub.md).
