# Adding Adapter Methods

This document describes how different efficient fine-tuning methods can be integrated into the codebase of `adapter-transformers`.
It can be used as a guide to add new efficient fine-tuning/ adapter methods.

Before we start to go into implementation details, first some important design philosophies of `adapter-transformers`:

- _Adapters should integrate seamlessly with existing model classes_: This means (a) if a model architecture supports adapters, it should be possible to use them with all model classes of this architecture and (b) adapters should be entirely opt-in, i.e. the model classes still must work without adapters.
- _Changes to the original should be minimal_: `adapter-transformers` tries to avoid changes to the original HF code as far as possible. We extensively use Python mixins to achieve this.

Now we highlight the most important components of integrating adapter methods into Transformer models.
Each integration is highly dependent on the specific details of the adapter methods.
Therefore, the described steps might not be applicable to each implementation.

## Implementation

❓ As adapter methods typically inject blocks of new parameters into an existing Transformer model, they mostly can be implemented using multiple blocks of classes deriving from `torch.nn.Module`.
These module classes then have to be inserted into the correct locations within the Transformer model implementation.
Thus, each adapter method implementation at least should provide two classes:

- a configuration class deriving from `AdapterConfigBase` that provides attributes for all configuration options of the method
- a module class deriving from the abstract `AdapterLayerBase` that provides the method parameters and a set of standard adapter management functions

**📝 Steps**

- All configuration classes reside in `src/transformers/adapters/configuration.py`.
    To add a new configuration class for a new method, create a new subclass of `AdapterConfigBase`.
    Make sure to set the `architecture` attribute in your class.
    - Finally, also make sure the config class is added to the `__init__.py` files in `src/transformers/adapters` and `src/transformers`.
- The `AdapterLayerBase` class from which any new adapter modules should derive resides in `src/transformers/adapters/layer.py`.
    - This abstract base class defines a set of methods that should be implemented by each deriving class,
    including methods for adding, enabling and deleting adapter weights.
    - Most importantly, the module classes deriving from this base class should implement the forward pass through an adaptation component.
    - The concrete implementation of these classes heavily depends on the specifics of the adapter method.
    For a reference implementation, have a look at `AdapterLayer` for bottleneck adapters.
- To actually make use of the newly implemented classes, it's finally necessary to integrate the forward calls to the modules in the actual model implementations.
    - This, again, is highly dependent on how the adapter method interacts with the base model classes Typically, module classes can be integrated either via mixins (see `src/transformers/adapters/mixins`) or directly as submodules of the respective model components.
    - The model class integration has to be repeated for each supported Transformer model, as they typically don't share a codebase.
    Please try to integrate any new adapter method into every model class when it's reasonable.
    You can find all currently supported model classes at https://docs.adapterhub.ml/model_overview.html.

**Additional things to consider**

- New adapter methods typically also require some changes in the `AdapterLoader` class in `src/transformers/adapters/loading.py` (also see [here](https://docs.adapterhub.ml/extending.html#loading-custom-module-weights)).
- Depending on the method to be integrated, further changes in other classes might be necessary.

## Testing

❓ `adapter-transformers` provides a framework for testing adapter methods on implementing models in `tests_adapters`.
Tests for each adapter method are provided via a mixin class.
All test mixins derive from the common `AdapterMethodBaseTestMixin` class and reside in `tests_adapters/methods`.

**📝 Steps**

- Add a new `test_<method>.py` module in `tests_adapters/methods`.
    - This module should contain a `<method>TestMixin` class deriving from `AdapterMethodBaseTestMixin` that implements typical methods of adding, loading and training modules of the new adapter method.
    - Have a look at existing test mixins for reference.
- Next, add the newly implemented test mixin to the tests of all model types that support the new adapter method.
    - Each model type has its own test class `tests_adapters/test_<model_type>.py` that contains a `<model_type>AdapterTest` class.
    Add the new test mixin to the mixins of this class.
    E.g., if the new method is supported by BERT, add the its test mixin to `BertAdapterTest`.

## Documentation

❓ The documentation for `adapter-transformers` lives in the `adapter_docs` folder.

**📝 Steps**

- Add the class documentation for the configuration class of the new method in `adapter_docs/classes/adapter_config.rst`.
- In `adapter_docs/overview.md`, add a new section for the new adapter method that describes the most important concepts. Please try to follow the general format of the existing methods.
- Add a new column in the table in `adapter_docs/model_overview.md` and check the models that support the new adapter method.

Finally, please add a row for the new method in the table of supported methods under _Implemented Methods_ in the main `README.md` of this repository.

## Training Example Adapters

❓ To make sure the new adapter implementation works properly, it is useful to train some example adapters and compare the training results to full model fine-tuning and/or reference implementations.
Ideally, this would include training adapters on one (or more) tasks that are good for demonstrating the new method and uploading them to AdapterHub.

HuggingFace already provides example training scripts for many tasks, some of them have already been modified to support adapter training (see https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples).
