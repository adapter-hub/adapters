# Testing the adapters library

This README gives an overview of how the test directory is organized and the possibilities to group and execute different kinds of tests.
## Test directory structure

```
tests/
├── __init__.py
├── fixtures/                               # Datasets, samples, ...
|   └── ...
├── test_impl/                              # Test Implementations
│   ├── __init__.py
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── test_adapter_composition.py
│   │   └── test_parallel.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_adapter_config.py
│   │   ├── test_adapter_conversion.py
│   │   └── ...
│   ├── embeddings/
│   └── ...
├── test_methods/                           # Test entry points 
│   └── __init__.py
├── test_models/                            # Test entry points 
│   └── __init__.py
```

## Test Types

1. Adapter method tests: test the **implementation of the adapter methods**, such as the different kind of adapters or costum heads.
    - These tests are exectued for each model, hence there is a testfile for each model, e.g. `test_albert.py`
    - Each model test file is organized in various test classes to group similar tests
    - While this results in a little bit more boilerplate code, it allows for an organized view in the test viewer, which in return also allows to conviniently execute subgroups of test, e.g. like this:
    ![alt text](image.png)
2. Adapter model tests: test the **implementation of the adapter models** on which the adapter methods can be used. 
    - We resort to the thorough test suite of Hugging Face and test our models on it.

## Utilizing pytest markers

Each class in each model test file in `tests/test_methods` is decorated with a marker of a certain type, e.g.:
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

These markers can be used to execute a certain type of test **for every model**:
- e.g.: for executing the compacter tests for every model we can write:
    ```bash
    cd tests/test_methods
    pytest -m lora
    ```
    This command will execute all lora tests for every model in the adapters libray

Alternatively to navigating to `tests/test_methods` in the terminal you can select a command from the `Makefile` in the root directory and launch such a subset of test via e.g.:
```bash
make test-adapter-lora
```