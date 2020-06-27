# Extending the Library

## Integrating new transformer models

Currently, the `adapter-transformers` library integrates adapter modules to language models base on `BERT`. However, the integration into new models is possible. `adapter-transformers` defines two abstract base classes that provide a generic adapter management setup:
- [`ModelAdaptersMixin`](classes/model_mixins.md#modeladaptersmixin)
- [`ModelWithHeadsAdaptersMixin`](classes/model_mixins.md#modelwithheadsadaptersmixin)

Depending on its type, every adapter-supporting model should derive from one of these mixin classes. `ModelAdaptersMixin` is intended for base model classes which do not provide task-specific prediction heads (e.g. `BertModel`, `RobertaModel`) whereas `ModelWithHeadsAdaptersMixin` should be used for models with prediction heads, such as `BertForSequenceClassification` or `BertModelWithHeads`.

Both classes specify a minimal number of abstract methods, such as `add_adapter()`, which must be implemented by every deriving model class. For the implementation details, refer to the [class documentation of the base classes](classes/model_mixins.md) or the [reference BERT implementation](classes/bert_mixins.md).

## Loading custom module weights

`adapter-transformers` provides support for saving and loading adapter and prediction head modules from the local file system or the Hub out of the box.
However, countless additional module integrations into language models are thinkable.
To provide a basis for such new custom model plugins, `adapter-transformers` integrates a basic mechanism to save and load custom weights.

All adapter and head module weights are extracted, saved and loaded by implementations of the [`WeightsLoader`](classes/weights_loaders.md#weightsloader) class, the two preincluded being [`AdapterLoader`](classes/weights_loaders.md#adapterloader) and [`PredictionHeadLoader`](classes/weights_loaders.md#predictionheadloader). To add basic saving and loading functionalities to your custom module weights, you can implement a new subclass of `WeightsLoader`. The two required abstract methods to be implemented are:

- `filter_func(self, name: str) -> Callable[[str], bool]`: The callable returned by this method is used to extract the module weights to be saved or loaded based on their names.

- `rename_func(self, old_name: str, new_name: str) -> Callable[[str], str]`: The callable returned by this method is used to optionally rename the module weights after loading.

For more advanced functionalities, you may also want to override the `save()` and `load()` method.

Using the custom loader class, weights can now be saved with:
```python
loader = MyCustomWeightsLoader(model)
loader.save("path/to/save/dir", "custom_weights_name")
```

You can also upload these weights to the Hub and then load them from there together with an adapter:
```python
model.load_adapter(
    "adapter_name",
    adapter_type="text_task",
    custom_weights_loaders=[MyCustomWeightsLoader]
)
```
