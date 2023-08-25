# Contributing to AdapterHub

There are many ways in which you can contribute to AdapterHub and the `adapter-transformers` library.
This includes code contributions such as:
- implementing new adapter methods
- adding support for new Transformer
- fixing open issues

as well as non-code contributions such as:
- training and uploading adapters to the Hub
- writing documentation and blog posts
- helping others with their issues and questions

Whichever way you'd like to contribute, you're very welcome to do so!

## Contributing to the `adapter-transformers` codebase

To get started with writing code for `adapter-transformers`, you'd want to set up the project on a local/ development environment.
`adapter-transformers` closely follows the original HuggingFace Transformers repository in many aspects.
As they already provide a great guide on setting up the project and the general contribution process, we refer to [their contributing guide](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) here.
Some additional notes are given below.

`adapter-transformers` uses the same code quality checks as HuggingFace Transformers.
Therefore, make sure to pass all the tests run using
```bash
$ make quality
```
to pass our CI pipeline.

Besides the commands for formatting, style checking and testing mentioned in the HuggingFace contributing guide, you can run all tests specific to `adapter-transformers` as follows:
```bash
$ make test-adapters
```
This corresponds to the tests run in our CI pipeline.

Below we refer to more detailed explanations of some typical contribution scenarios.

### Adding Adapter Methods

How to integrate new efficient fine-tuning/ adapter methods to `adapter-transformers` is described at [https://docs.adapterhub.ml/contributing/adding_adapter_methods.html](https://docs.adapterhub.ml/contributing/adding_adapter_methods.html).

### Adding Adapters to a Model

How to add adapter support to a model type already supported by HuggingFace Transformers is described at [https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html](https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html).

## Contributing Adapters to the Hub

How to make your own trained adapters accessible via AdapterHub is described at [https://docs.adapterhub.ml/hub_contributing.html](https://docs.adapterhub.ml/hub_contributing.html).
