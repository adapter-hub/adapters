# Adding Adapters to a Model

This document gives an overview on how `adapter-transformers` integrates adapter modules into the model architectures of HuggingFace Transformers.
It can be used as a guide to add adapter support to new model architectures.

Before we start to go into implementation details, first some important design philosophies of `adapter-transformers`:

- _Adapters should integrate seamlessly with existing model classes_: This means (a) if a model architecture supports adapters, it should be possible to use them with all model classes of this architecture and (b) adapters should be entirely opt-in, i.e. the model classes still must work without adapters.
- _Changes to the original should be minimal_: `adapter-transformers` tries to avoid changes to the original HF code as far as possible. We extensively use Python mixins to achieve this.

Now we go through the integration of adapters into an existing model architecture step by step.

**The following steps might not be applicable to every model architecture.**

## Implementation

❓ Each model architecture with adapter support has a main `<model_type>.py` module in `src/transformers/adapters/models` (e.g. `src/transformers/adapters/models/distilbert.py` for `modeling_distilbert.py`) that provides the required adapter mixins for each modeling component (e.g. there is a `DistilBertTransfomerBlockAdaptersMixin` for the `TransformerBlock` of DistilBERT etc.).
This is the central module to implement.

**📝 Steps**

- Add a new `<model_type>.py` module for your architecture in `src/transformers/adapters/models` (or reuse an existing if possible).
    - There usually should be one mixin that derives from `AdapterLayerBaseMixin` or has it as a child module.
    - The mixin for the whole base model class (e.g. `BertModel`) should derive from `ModelAdaptersMixin` and (if possible) `InvertibleAdaptersMixin`. Make sure to implement the abstract methods these mixins might define.
    - Have a look at existing examples, e.g. `distilbert.py`, `bert.py`.
- Implement the mixins on the modeling classes (`modeling_<model_type>.py`).
    - Make sure the calls to `adapters_forward()` are added in the right places.
    - The base model class (e.g. `BertModel`) should implement the mixin derived from `ModelAdaptersMixin` you created previously.
    - The model classes with heads (e.g. `BertForSequenceClassification`) should directly implement `ModelWithHeadsAdaptersMixin`.
- Add the mixin for config classes, `ModelConfigAdaptersMixin`, to the model configuration class in `configuration_<model_type>`.
    - There are some naming differences on the config attributes of different model architectures. The adapter implementation requires some additional attributes with a specific name to be available. These currently are `hidden_dropout_prob` and `attention_probs_dropout_prob` as in the `BertConfig` class.

❓ Adapter-supporting architectures have a new model class `<model_type>ModelWithHeads`.
These classes allow flexible adding of and switching between multiple prediction heads of different types.

**📝 Steps**

- In `modeling_<model_type>.py`, add a new `<model_type>ModelWithHeads` class.
    - This class should implement a mixin (in `src/transformers/adapters/models/<model_type>.py`) which derives from `ModelWithFlexibleHeadsAdaptersMixin`
    - In the mixin, add methods for those prediction heads that make sense for the new model architecture.
- Add `<model_type>ModelWithHeads` to the `MODEL_WITH_HEADS_MAPPING` mapping in `modeling_auto.py` and to `__init__.py`.

### Additional (optional) implementation steps

- Dynamic adapter activation via `adapter_names` argument (cf. [PR#176](https://github.com/Adapter-Hub/adapter-transformers/pull/176)).
- Parallel adapter inference via `Parallel` composition block (cf. [documentation](https://docs.adapterhub.ml/adapter_composition.html#parallel), [PR#150](https://github.com/Adapter-Hub/adapter-transformers/pull/150)).
- Provide mappings for an architecture's existing (static) prediction heads into `adapter-transformers` flex heads (cf. [implementation](https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/head_utils.py#L8)).

## Testing

❓ In addition to the general HuggingFace model tests, there are adapter-specific test cases (usually starting with `test_adapter_`).

**📝 Steps**

- Add a new `<model_type>AdapterTest` class in `test_adapter.py` similar to the existing classes (e.g. `BertAdapterTest`).
- Add `<model_type>ModelWithHeads` to `test_modeling_<model_type>.py`.
- Insert `test_modeling_<model_type>` into the list of tested modules in `utils/run_tests.py`.
- Append `<model_type>` to the list in `check_adapters.py`.

## Documentation

❓ The documentation for `adapter-transformers` lives in the `adapter_docs` folder.

**📝 Steps**

- Add `adapter_docs/classes/models/<model_type>.rst` (oriented at the doc file in the HF docs, make sure to include `<model_type>ModelWithHeads` and the HF notice). 
Finally, list the file in `index.rst`.

## Training Example Adapters

❓ To make sure the new adapter implementation works properly, it is useful to train some example adapters and compare the training results to full model fine-tuning. Ideally, this would include training adapters on one (or more) tasks that are good for demonstrating the new model architecture (e.g. GLUE benchmark for BERT, summarization for BART) and uploading them to AdapterHub.

HuggingFace already provides example training scripts for many tasks, some of them have already been modified to support adapter training (see https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples).
