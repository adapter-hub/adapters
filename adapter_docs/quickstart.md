# Quickstart

## Introduction

Currently, *adapter-transformers* adds adapter components to all transformer models based on BERT and using PyTorch.
For working with adapters, a couple of methods for creation (e.g. `add_adapter()`), loading (e.g. `load_adapter()`) and
storing (e.g. `save_adapter()`) are added to the model classes. In the following, we will briefly go through some examples.

```eval_rst
.. note::
    This document focuses on the adapter-related functionalities added by *adapter-transformers*.
    For a more general overview of the *transformers* library, visit
    `the 'Usage' section in Huggingface's documentation <https://huggingface.co/transformers/usage.html>`_.
```

## Quick Tour: Using a pre-trained adapter for inference

_We also have a Quickstart Colab notebook for adapter inference:_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/website/blob/master/app/static/notebooks/Adapter_Quickstart_Inference.ipynb)

The following example shows the usage of a basic pre-trained transformer model with adapters.
Our goal here is to predict the sentiment of a given sentence.

We use BERT in this example, so we first load a pre-trained `BertTokenizer` to encode the input sentence and a pre-trained
`BertModel` from HuggingFace:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# output more information
import logging
logging.basicConfig(level=logging.INFO)

# load pre-trained BERT tokenizer from Huggingface
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize an input sentence
sentence = "It's also, clearly, great fun."

# convert input tokens to indices and create PyTorch input tensor
input_tensor = torch.tensor([tokenizer.encode(sentence)])

# load pre-trained BERT model from Huggingface
# the `BertForSequenceClassification` class includes a prediction head for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

Having loaded the model, we now add a pre-trained task adapter that is useful to our task from Adapter Hub.
As we're doing sentiment classification, we use [an adapter trained on the SST-2 dataset](https://adapterhub.ml/adapters/ukp/bert-base-uncased-sst_pfeiffer/) in this case.
The task prediction head loaded together with the adapter gives us a class label for our sentence:

```python
# load pre-trained task adapter from Adapter Hub
# this method call will also load a pre-trained classification head for the adapter task
adapter_name = model.load_adapter('sst-2@ukp', config='pfeiffer')

# activate the adapter we just loaded, so that it is used in every forward pass
model.set_active_adapters(adapter_name)

# predict output tensor
outputs = model(input_tensor)

# retrieve the predicted class label
predicted = torch.argmax(outputs[0]).item()
assert predicted == 1
```

To save our pre-trained model and adapters, we can easily store and reload them as follows:

```python
# save model
model.save_pretrained('./path/to/model/directory/')
# save adapter
model.save_adapter('./path/to/adapter/directory/', 'sst-2')

# load model
model = BertModel.from_pretrained('./path/to/model/directory/')
model.load_adapter('./path/to/adapter/directory/')
```

Similar to how the weights of the full model are saved, the `save_adapter()` will create a file for saving the adapter weights and a file for saving the adapter configuration in the specified directory.

## Quick Tour: Adapter training

_We also have a Quickstart Colab notebook for adapter training:_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/website/blob/master/app/static/notebooks/Adapter_Quickstart_Training.ipynb)

For more examples on training different adapter setups, refer to the section on [Adapter Training](training.md).
