# Quickstart

## Introduction

Currently, *adapter-transformers* adds adapter components to the PyTorch implementations of all transformer models listed in the *Supported Models* section.
For working with adapters, a couple of methods for creation (`add_adapter()`), loading (`load_adapter()`), 
storing (`save_adapter()`) and deletion (`delete_adapter()`) are added to the model classes. In the following, we will briefly go through some examples.

```{eval-rst}
.. note::
    This document focuses on the adapter-related functionalities added by *adapter-transformers*.
    For a more general overview of the *transformers* library, visit
    `the 'Usage' section in HuggingFace's documentation <https://huggingface.co/transformers/usage.html>`_.
```

## Quick Tour: Using a pre-trained adapter for inference

_We also have a Quickstart Colab notebook for adapter inference:_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/02_Adapter_Inference.ipynb)

The following example shows the usage of a basic pre-trained transformer model with adapters.
Our goal here is to predict the sentiment of a given sentence.

We use BERT in this example, so we first load a pre-trained `BertTokenizer` to encode the input sentence and a pre-trained
`bert-base-uncased` checkpoint from HuggingFace's Model Hub using the [`BertAdapterModel`](transformers.adapters.BertAdapterModel) class:

```python
import os

import torch
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel, AutoAdapterModel

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# An input sentence
sentence = "It's also, clearly, great fun."

# Tokenize the input sentence and create a PyTorch input tensor
input_data = tokenizer(sentence, return_tensors="pt")

# Load pre-trained BERT model from HuggingFace Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
# It can be used with different prediction heads
model = BertAdapterModel.from_pretrained('bert-base-uncased')
```

Having loaded the model, we now add a pre-trained task adapter that is useful to our task from AdapterHub.
In this case, for sentiment classification, we thus use [an adapter trained on the SST-2 dataset](https://adapterhub.ml/adapters/ukp/bert-base-uncased_sentiment_sst-2_pfeiffer/).
The task prediction head loaded together with the adapter gives us a class label for our sentence:

```python
# Load pre-trained task adapter from Adapter Hub
# This method call will also load a pre-trained classification head for the adapter task
adapter_name = model.load_adapter("sentiment/sst-2@ukp", config='pfeiffer')

# Activate the adapter we just loaded, so that it is used in every forward pass
model.set_active_adapters(adapter_name)

# Predict output tensor
outputs = model(**input_data)

# Retrieve the predicted class label
predicted = torch.argmax(outputs[0]).item()
assert predicted == 1
```

To save our pre-trained model and adapters, we can easily store and reload them as follows:

```python
# For the sake of this demonstration an example path for loading and storing is given below
example_path = os.path.join(os.getcwd(), "adapter-quickstart")

# Save model
model.save_pretrained(example_path)
# Save adapter
model.save_adapter(example_path, adapter_name)

# Load model, similar to HuggingFace's AutoModel class, 
# you can also use AutoAdapterModel instead of BertAdapterModel
model = AutoAdapterModel.from_pretrained(example_path)
model.load_adapter(example_path)
```

Similar to how the weights of the full model are saved, the `save_adapter()` will create a file for saving the adapter weights and a file for saving the adapter configuration in the specified directory.

Finally, if we have finished working with adapters, we can restore the base Transformer to its original form by deactivating and deleting the adapter:

```python
# Deactivate all adapters
model.set_active_adapters(None)
# Delete the added adapter
model.delete_adapter(adapter_name)
```

## Quick Tour: Adapter training

_We also have a Quickstart Colab notebook for adapter training:_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb)
For more examples on training different adapter setups, refer to the section on [Adapter Training](training.md).
