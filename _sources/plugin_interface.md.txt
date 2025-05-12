# Custom Models

The _Adapters_ library provides a simple mechanism for integrating adapter methods into any available _Transformers_ model - including custom architectures. While most adapter methods are supported through this interface, some features like Prefix Tuning are not available.

## Pre-supported Models

Some models already have interfaces provided by default in the library. For these models, you can simply initialize the model using adapters without specifying an interface:

```python
import adapters
from transformers import AutoModelForMaskedLM 

model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")  
adapters.init(model)
```

Check out our [Model Overview](model_overview.html) page to see all models that are supported out of the box.

## Adding Support for New Models

If we don't support your model yet, you can easily add adapter support by defining a plugin interface instance of [`AdapterModelInterface`](adapters.AdapterModelInterface). Here's an example for Gemma 2:

```python
import adapters
from adapters import AdapterModelInterface
from transformers import AutoModelForCausalLM

plugin_interface = AdapterModelInterface(
    adapter_methods=["lora", "reft"],
    model_embeddings="embed_tokens",
    model_layers="layers",
    layer_self_attn="self_attn",
    layer_cross_attn=None,
    attn_k_proj="k_proj",
    attn_q_proj="q_proj",
    attn_v_proj="v_proj",
    attn_o_proj="o_proj",
    layer_intermediate_proj="mlp.up_proj",
    layer_output_proj="mlp.down_proj",
)

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", token="<YOUR_TOKEN>")
adapters.init(model, interface=plugin_interface)

model.add_adapter("my_adapter", config="lora")

print(model.adapter_summary())
```

### Contributing Interfaces

We encourage you to share your adapter interfaces with the community! The interfaces for all pre-supported models can be found in our [`interfaces.py`](https://github.com/adapter-hub/adapters/blob/main/src/adapters/wrappers/interfaces.py) file. You can:
- Open a pull request to add your interface directly
- If you're short on time, just drop your interface in a GitHub issue and we'll add it for you

### Walkthrough

Let's go through what happens in the example above step by step:

**1. Define adapter methods to plug into a model:**  
The `adapter_methods` argument is the central parameter to configure which adapters will be supported in the model.
Here, we enable all LoRA and ReFT based adapters.
See [`AdapterMethod`](adapters.AdapterMethod) for valid options to specify here.
Check out [Adapter Methods](methods.md) for detailed explanation of the methods.

**2. Define layer and module names:**  
While all Transformers layers share similar basic components, their implementation can differ in terms of subtleties such as module names.
Therefore, the [`AdapterModelInterface`](adapters.AdapterModelInterface) needs to translate the model-specific module structure into a common set of access points for adapter implementations to hook in.
The remaining attributes in the definition above serve this purpose.
Their attribute names follow a common syntax that specify their location and purpose:
- The initial part before the first "_" defines the base module relative to which the name should be specified.
- The remaining part after the first "_" defines the functional component.

E.g., `model_embeddings` identifies the embeddings layer (functional component) relative to the base model (location).
`layer_output_proj` identifies the FFN output projection relative to one Transformer layer.
Each attribute value may specify a direct submodule of the reference module (`"embed_token"`) or a multi-level path starting at the reference module (`"mlp.down_proj"`).

**3. (optional) Extended interface attributes:**  
There are a couple of attributes in the [`AdapterModelInterface`](adapters.AdapterModelInterface) that are only required for some adapter methods.
We don't need those in the above example for LoRA and ReFT, but when supporting bottleneck adapters as well, the full interface would look as follows:
```python
adapter_interface = AdapterModelInterface(
    adapter_types=["bottleneck", "lora", "reft"],
    model_embeddings="embed_tokens",
    model_layers="layers",
    layer_self_attn="self_attn",
    layer_cross_attn=None,
    attn_k_proj="k_proj",
    attn_q_proj="q_proj",
    attn_v_proj="v_proj",
    attn_o_proj="o_proj",
    layer_intermediate_proj="mlp.up_proj",
    layer_output_proj="mlp.down_proj",
    layer_pre_self_attn="input_layernorm",
    layer_pre_cross_attn=None,
    layer_pre_ffn="pre_feedforward_layernorm",
    layer_ln_1="post_attention_layernorm",
    layer_ln_2="post_feedforward_layernorm",
)
```

**4. Initialize adapter methods in the model:**
Finally, we just need to apply the defined adapter integration in the target model.
This can be achieved using the usual `adapters.init()` method:
```python
adapters.init(model, interface=adapter_interface)
```
Now, you can use (almost) all functionality of the _Adapters_ library on the adapted model instance!


```{eval-rst}
.. note::
    Some models like GPT-2 or ModernBERT have the query, value and key layer in one single tensor. In this case, you must set the `attn_qkv_proj` instead of setting `attn_k_proj`, `attn_q_proj` and `attn_v_proj`.
```

## Limitations

The following features of the _Adapters_ library are not supported via the plugin interface approach:
- Prefix Tuning adapters
- Parallel composition blocks
- XAdapterModel classes
- Setting `original_ln_after=False` in bottleneck adapter configurations (this affects `AdapterPlusConfig`)
