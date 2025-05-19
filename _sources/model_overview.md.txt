# Model Overview

This page gives an overview of the Transformer models currently supported by `adapters`.
The table below further shows which model architectures support which adaptation methods and which features of `adapters`.

```{eval-rst}
.. note::
    Each supported model architecture X typically provides a class ``XAdapterModel`` for usage with ``AutoAdapterModel``.
    Additionally, it is possible to use adapters with the model classes already shipped with Hugging Face Transformers. For these classes, initialize the model for adapters with `adapters.init(model)`.
    E.g., for BERT, this means adapters provides a ``BertAdapterModel`` class, but you can also use ``BertModel``, ``BertForSequenceClassification`` etc. together with adapters.
```


| Model                                   | (Bottleneck)<br> Adapters | Prefix<br> Tuning | LoRA | Compacter | Adapter<br> Fusion | Invertible<br> Adapters | Parallel<br> block | Prompt<br> Tuning | ReFT |
| --------------------------------------- | -| - | - | - | - | - | - |- | - |
| [Custom models](plugin_interface.html)<sup>◊</sup> | ✅ |  | ✅ | ✅ | ✅ | ✅ |  | ✅ | ✅ |
| [ALBERT](classes/models/albert.html)    | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [BART](classes/models/bart.html)        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [BEIT](classes/models/beit.html)        | ✅ | ✅ | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |
| [BERT-Generation](classes/models/bert-generation.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [BERT](classes/models/bert.html)        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [CLIP](classes/models/clip.html)        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |
| [DeBERTa](classes/models/deberta.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [DeBERTa-v2](classes/models/debertaV2.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [DistilBERT](classes/models/distilbert.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Electra](classes/models/electra.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Encoder Decoder](classes/models/encoderdecoder.html) | (*) | (*) | (*) | (*) | (*) | (*) | | | (*) |
| Gemma 2<sup>◊</sup>                    | ✅ |  | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |
| Gemma 3 (Text)<sup>◊</sup>                    | ✅ |  | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |
| [GPT-2](classes/models/gpt2.html)       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [GPT-J](classes/models/gptj.html)       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [Llama<br>Llama 2<br>Llama 3](classes/models/llama.html)       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [MBart](classes/models/mbart.html)      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| ModernBERT<sup>◊</sup>                 | ✅ |  | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |
| [Mistral](classes/models/mistral.html)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [MT5](classes/models/mt5.html)          | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| Phi-1<br>Phi-2<sup>◊</sup>                 | ✅ |  | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |
| [PLBart](classes/models/plbart.html)      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| Qwen2<br>Qwen2.5<br>Qwen3 <sup>◊</sup>                    | ✅ |  | ✅ | ✅ | ✅ | ✅ |  |  | ✅ |
| [RoBERTa](classes/models/roberta.html)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [T5](classes/models/t5.html)            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [ViT](classes/models/vit.html)            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Whisper](classes/models/whisper.html)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [XLM-RoBERTa](classes/models/xlmroberta.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [X-MOD](classes/models/xmod.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

<sup>◊</sup> Models supported via the [plugin interface mechanism](plugin_interface.html). `original_ln_after=False` is unsupported for bottleneck configs.  
(*) If the used encoder and decoder model class are supported.


**Missing a model architecture you'd like to use?**  
The new model plugin interface makes it easy to support new transformer models with just a few lines of code [Learn more](plugin_interface.md).
Also, _Adapters_ can be extended to new model architectures as described in [Adding Adapters to a Model](https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html).
Feel free to [open an issue](https://github.com/Adapter-Hub/adapters/issues) requesting support for a new architecture.
_We very much welcome pull requests adding new model implementations!_
