# The adapter-transformers documentation

This is the documentation of the adapter-related parts of the transformers library and the Adapter-Hub. Huggingface's documentation of the base library is located in the `/docs` folder.

## Installing & Building

Building the documentation requires some additional packages installed. You can install them by running the following command in the root folder:

```bash
pip install -e ".[docs]"
```

Cleaning and regenerating the documentation files can be done using `sphinx` by running the following command in the `/adapter_docs` folder:

```bash
make clean && make html
```

The build output will be located in `/adapter_docs/_build/html`.
