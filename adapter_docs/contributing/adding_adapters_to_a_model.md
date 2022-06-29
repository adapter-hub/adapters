# Adding Adapters to a Model

This document gives an overview on how `adapter-transformers` integrates adapter modules into the model architectures of HuggingFace Transformers.
It can be used as a guide to add adapter support to new model architectures.

Before we start to go into implementation details, first some important design philosophies of `adapter-transformers`:

- _Adapters should integrate seamlessly with existing model classes_: This means (a) if a model architecture supports adapters, it should be possible to use them with all model classes of this architecture and (b) adapters should be entirely opt-in, i.e. the model classes still must work without adapters.
- _Changes to the original should be minimal_: `adapter-transformers` tries to avoid changes to the original HF code as far as possible. We extensively use Python mixins to achieve this.

Now we go through the integration of adapters into an existing model architecture step by step.

**The following steps might not be applicable to every model architecture.**

## Implementation

### Integration into model implementation

❓ Adding adapter support to an existing model architecture requires modifying a few parts of the model forward pass logic. These changes have to be made directly in the respective `modeling_<model_type>.py` class.
Additionally, a few adapter mixins need to be applied to the respective Transformer module classes to provide the adapter implementations to a model.
For this purpose, there typically exists a module `src/transformers/adapters/mixins/<model_type>.py`.

**📝 Steps**

- Add a new `<model_type>.py` module for your architecture in `src/transformers/adapters/mixins` (or reuse an existing if possible).
    - There usually exists a mixin on the Transformer layer level that derives that holds modules for adapter layers.
    - The mixin for the whole base model class (e.g. `BertModel`) should derive from `ModelAdaptersMixin` and (if possible) `EmbeddingAdaptersMixin` and/or `InvertibleAdaptersMixin`. This mixin should at least implement the `iter_layers()` method but might require additional modifications depending on the architecture.
    - Have a look at existing examples, e.g. `distilbert.py`, `bert.py`.
- Implement the mixins and the required modifications on the modeling classes (`modeling_<model_type>.py`).
    - Make sure the calls to `adapter_layer_forward()` are added in the right places.
    - The base model class (e.g. `BertModel`) should implement the mixin derived from `ModelAdaptersMixin` you created previously.
    - The model classes with heads (e.g. `BertForSequenceClassification`) should directly implement `ModelWithHeadsAdaptersMixin`.
    - To additionally support Prefix Tuning, it's necessary to apply the forward call to the `PrefixTuningShim` module in the respective attention layer.
    - Again, have a look at existing implementations, e.g. `modeling_distilbert.py` or `modeling_bart.py`.
- Adapt the config class to the requirements of adapters in `src/transformers/adapters/wrappers/configuration.py`.
    - There are some naming differences on the config attributes of different model architectures. The adapter implementation requires some additional attributes with a specific name to be available. These currently are `num_attention_heads`, `hidden_size`, `hidden_dropout_prob` and `attention_probs_dropout_prob` as in the `BertConfig` class.
    If your model config does not provide these, add corresponding mappings to `CONFIG_CLASS_KEYS_MAPPING`.

### `...AdapterModel` class

❓ Adapter-supporting architectures should provide a new model class `<model_type>AdapterModel`.
This class allows flexible adding of and switching between multiple prediction heads of different types.

**📝 Steps**

- In `src/transformers/adapters/models`, add a new `<model_type>.py` file.
    - This module should implement the `<model_type>AdapterModel` class, deriving from `ModelWithFlexibleHeadsAdaptersMixin` and `<model_type>PreTrainedModel`.
    - In the model class, add methods for those prediction heads that make sense for the new model architecture.
    - Again, have a look at existing implementations, e.g. `bert.py`. Note that the `<model_type>ModelWithHeads` classes in existing modules are kept for backwards compatibility and are not needed for newly added architectures.
- Add `<model_type>AdapterModel` to the `ADAPTER_MODEL_MAPPING_NAMES` mapping in `src/transformers/adapters/models/auto.py` and to `src/transformers/adapters/__init__.py`.

### Additional (optional) implementation steps

- Parallel adapter inference via `Parallel` composition block (cf. [documentation](https://docs.adapterhub.ml/adapter_composition.html#parallel), [PR#150](https://github.com/Adapter-Hub/adapter-transformers/pull/150)).
- Provide mappings for an architecture's existing (static) prediction heads into `adapter-transformers` flex heads (cf. [implementation](https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/head_utils.py#L8)).

## Testing

❓ In addition to the general HuggingFace model tests, there are adapter-specific test cases. All tests are executed from the `tests_adapters` folder.

**📝 Steps**

- Add a new `test_<model_type>.py` module in `tests_adapters`. This module typically holds three test classes:
    - `<model_type>AdapterModelTest` derives directly from HuggingFace's existing model test class `<model_type>ModelTest` and adds `<model_type>AdapterModel` as class to test.
    - `<model_type>AdapterModelTest` derives from a collection of test mixins that hold various adapter tests (depending on the implementation).
    - (optionally) `<model_type>ClassConversionTest` runs tests for correct class conversion if conversion of prediction heads is implemented.
- Append `<model_type>` to the list in `check_adapters.py`.

## Documentation

❓ The documentation for `adapter-transformers` lives in the `adapter_docs` folder.

**📝 Steps**

- Add `adapter_docs/classes/models/<model_type>.rst` (oriented at the doc file in the HF docs). Make sure to include `<model_type>AdapterModel` autodoc. Finally, list the file in `index.rst`.
- Add a new row for the model in the model table of the overview page at `adapter_docs/model_overview.md`, listing all the methods implemented by the new model.

## Training Example Adapters

❓ To make sure the new adapter implementation works properly, it is useful to train some example adapters and compare the training results to full model fine-tuning. Ideally, this would include training adapters on one (or more) tasks that are good for demonstrating the new model architecture (e.g. GLUE benchmark for BERT, summarization for BART) and uploading them to AdapterHub.

HuggingFace already provides example training scripts for many tasks, some of them have already been modified to support adapter training (see https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples).
