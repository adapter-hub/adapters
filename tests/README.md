# Testing the adapters library

This README gives an overview of how the test directory is organized and the possibilities of grouping and executing different kinds of tests.
## Overview test directory structure

```
tests/
├── __init__.py
├── fixtures/                       # Datasets, test samples, ...
|   └── ...
├── test_methods/                   # Dynamic adapter method tests (all models)
│   ├── __init__.py
│   ├── method_test_impl/               # Implementation of tests
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── composition/
│   │   └── ...
│   ├── base.py                         # Base from which model test bases inherit from                           
│   ├── generator.py                    # Testcase generation and registration
│   ├── test_albert.py                  # Example model test base testing adapter methods on the model
│   ├── test_beit.py 
│   └── ...
├── test_misc/                      # Miscellaneous adapter method tests (single model)
│   ├── test_adapter_config.py 
│   └── ...
├── test_models/                    # Adapter model tests with Hugging Face test suite
│   └── __init__.py
│   │   ├── base.py
│   │   ├── test_albert.py
│   │   └── ...
```

We differentiate between three kinds of tests:

1. Dynamic adapter method tests: These tests cover most functionalities of the adapters library, e.g. the individual adapter methods (LoRA, prompt tuning) or head functionalities and **are executed on every model**
2. Miscellaneous adapter method tests: These are the remaining tests not covered by the dynamic tests and are **only executed on a single model** to spare ressources as repeated execution on every model would not provide additional value
3. Adapter model tests: These tests **check the implementation of the adapter models** themselves, by applying the Hugging Face model test suite

## Test Generator $ Pytest Markers

This chapter zooms in on the test_methods directory. The main actor here is the file `generator.py` which is used by every model test base to generate the appropriate set of adapter method tests. Those tests are then registered in the respective model test file, like this:

``` python
method_tests = generate_method_tests(AlbertAdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
```

Each generatable class in `tests/test_methods` is decorated with a marker of a certain type, e.g.:
``` python
@require_torch
@pytest.mark.lora
class LoRA(
    AlbertAdapterTestBase,
    LoRATestMixin,
    unittest.TestCase,
):
    pass
```

These markers can be used to execute a certain type of test **for every model**. To use them you have two options:
1. Use `make` command:
    ```bash
    make test-adapter-method-subset subset=lora
    ```

2. Navigate to directory and directly execute:
    ```bash
    cd tests/test_methods
    pytest -m lora
    ```
Both versions will execute all LoRA tests for every model in the adapters library.
