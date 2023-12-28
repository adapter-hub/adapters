<!---
Copyright 2020 The AdapterHub Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

> **Note**: This repository holds the codebase of the _Adapters_ library, which has replaced `adapter-transformers`. For the legacy codebase, go to: https://github.com/adapter-hub/adapter-transformers-legacy.

<p align="center">
<img style="vertical-align:middle" src="https://raw.githubusercontent.com/Adapter-Hub/adapters/main/docs/logo.png" />
</p>
<h1 align="center">
<span><i>Adapters</i></span>
</h1>

<h3 align="center">
A Unified Library for Parameter-Efficient and Modular Transfer Learning
</h3>

![Tests](https://github.com/Adapter-Hub/adapters/workflows/Tests/badge.svg?branch=adapters)
[![GitHub](https://img.shields.io/github/license/adapter-hub/adapters.svg?color=blue)](https://github.com/adapter-hub/adapters/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/adapters)](https://pypi.org/project/adapters/)

`adapters` is an add-on to [HuggingFace's Transformers](https://github.com/huggingface/transformers) library, integrating adapters into state-of-the-art language models by incorporating **[AdapterHub](https://adapterhub.ml)**, a central repository for pre-trained adapter modules.

## Installation

`adapters` currently supports **Python 3.8+** and **PyTorch 1.10+**.
After [installing PyTorch](https://pytorch.org/get-started/locally/), you can install `adapters` from PyPI ...

```
pip install -U adapters
```

... or from source by cloning the repository:

```
git clone https://github.com/adapter-hub/adapters.git
cd adapters
pip install .
```

## Quick Tour

#### Load pre-trained adapters:

```python
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

model = AutoAdapterModel.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model.load_adapter("AdapterHub/roberta-base-pf-imdb", source="hf", set_active=True)

print(model(**tokenizer("This works great!", return_tensors="pt")).logits)
```

**[Learn More](https://docs.adapterhub.ml/loading.html)**

#### Adapt existing model setups:

```python
import adapters
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("t5-base")

adapters.init(model)

model.add_adapter("my_lora_adapter", config="lora")
model.train_adapter("my_lora_adapter")

# Your regular training loop...
```

**[Learn More](https://docs.adapterhub.ml/quickstart.html)**

#### Flexibly configure adapters:

```python
from adapters import ConfigUnion, PrefixTuningConfig, ParBnConfig, AutoAdapterModel

model = AutoAdapterModel.from_pretrained("microsoft/deberta-v3-base")

adapter_config = ConfigUnion(
    PrefixTuningConfig(prefix_length=20),
    ParBnConfig(reduction_factor=4),
)
model.add_adapter("my_adapter", config=adapter_config, set_active=True)
```

**[Learn More](https://docs.adapterhub.ml/overview.html)**

#### Easily compose adapters in a single model:

```python
from adapters import AdapterSetup, AutoAdapterModel
import adapters.composition as ac

model = AutoAdapterModel.from_pretrained("roberta-base")

qc = model.load_adapter("AdapterHub/roberta-base-pf-trec")
sent = model.load_adapter("AdapterHub/roberta-base-pf-imdb")

with AdapterSetup(ac.Parallel(qc, sent)):
    print(model(**tokenizer("What is AdapterHub?", return_tensors="pt")))
```

**[Learn More](https://docs.adapterhub.ml/adapter_composition.html)**

## Useful Resources

HuggingFace's great documentation on getting started with _Transformers_ can be found [here](https://huggingface.co/transformers/index.html). `adapters` is fully compatible with _Transformers_.

To get started with adapters, refer to these locations:

- **[Colab notebook tutorials](https://github.com/Adapter-Hub/adapters/tree/main/notebooks)**, a series notebooks providing an introduction to all the main concepts of (adapter-)transformers and AdapterHub
- **https://docs.adapterhub.ml**, our documentation on training and using adapters with _adapters_
- **https://adapterhub.ml** to explore available pre-trained adapter modules and share your own adapters
- **[Examples folder](https://github.com/Adapter-Hub/adapters/tree/main/examples/pytorch)** of this repository containing HuggingFace's example training scripts, many adapted for training adapters

## Implemented Methods

Currently, adapters integrates all architectures and methods listed below:

| Method | Paper(s) | Quick Links |
| --- | --- | --- |
| Bottleneck adapters | [Houlsby et al. (2019)](https://arxiv.org/pdf/1902.00751.pdf)<br> [Bapna and Firat (2019)](https://arxiv.org/pdf/1909.08478.pdf) | [Quickstart](https://docs.adapterhub.ml/quickstart.html), [Notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb) |
| AdapterFusion | [Pfeiffer et al. (2021)](https://aclanthology.org/2021.eacl-main.39.pdf) | [Docs: Training](https://docs.adapterhub.ml/training.html#train-adapterfusion), [Notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/03_Adapter_Fusion.ipynb) |
| MAD-X,<br> Invertible adapters | [Pfeiffer et al. (2020)](https://aclanthology.org/2020.emnlp-main.617/) | [Notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/04_Cross_Lingual_Transfer.ipynb) |
| AdapterDrop | [Rücklé et al. (2021)](https://arxiv.org/pdf/2010.11918.pdf) | [Notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/05_Adapter_Drop_Training.ipynb) |
| MAD-X 2.0,<br> Embedding training | [Pfeiffer et al. (2021)](https://arxiv.org/pdf/2012.15562.pdf) | [Docs: Embeddings](https://docs.adapterhub.ml/embeddings.html), [Notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/08_NER_Wikiann.ipynb) |
| Prefix Tuning | [Li and Liang (2021)](https://arxiv.org/pdf/2101.00190.pdf) | [Docs](https://docs.adapterhub.ml/methods.html#prefix-tuning) |
| Parallel adapters,<br> Mix-and-Match adapters | [He et al. (2021)](https://arxiv.org/pdf/2110.04366.pdf) | [Docs](https://docs.adapterhub.ml/method_combinations.html#mix-and-match-adapters) |
| Compacter | [Mahabadi et al. (2021)](https://arxiv.org/pdf/2106.04647.pdf) | [Docs](https://docs.adapterhub.ml/methods.html#compacter) |
| LoRA | [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf) | [Docs](https://docs.adapterhub.ml/methods.html#lora) |
| (IA)^3 | [Liu et al. (2022)](https://arxiv.org/pdf/2205.05638.pdf) | [Docs](https://docs.adapterhub.ml/methods.html#ia-3) |
| UniPELT | [Mao et al. (2022)](https://arxiv.org/pdf/2110.07577.pdf) | [Docs](https://docs.adapterhub.ml/method_combinations.html#unipelt) |
| Prompt Tuning | [Lester et al. (2021)](https://aclanthology.org/2021.emnlp-main.243/) | [Docs](https://docs.adapterhub.ml/methods.html#prompt-tuning) |

## Supported Models

We currently support the PyTorch versions of all models listed on the **[Model Overview](https://docs.adapterhub.ml/model_overview.html) page** in our documentation.

## Developing & Contributing

To get started with developing on _Adapters_ yourself and learn more about ways to contribute, please see https://docs.adapterhub.ml/contributing.html.

## Citation

If you use _Adapters_ in your work, please consider citing our library paper: [Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning](https://arxiv.org/abs/2311.11077)

```
@inproceedings{poth-etal-2023-adapters,
    title = "Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning",
    author = {Poth, Clifton  and
      Sterz, Hannah  and
      Paul, Indraneil  and
      Purkayastha, Sukannya  and
      Engl{\"a}nder, Leon  and
      Imhof, Timo  and
      Vuli{\'c}, Ivan  and
      Ruder, Sebastian  and
      Gurevych, Iryna  and
      Pfeiffer, Jonas},
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-demo.13",
    pages = "149--160",
}
```

Alternatively, for the predecessor `adapter-transformers`, the Hub infrastructure and adapters uploaded by the AdapterHub team, please consider citing our initial paper: [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779)

```
@inproceedings{pfeiffer2020AdapterHub,
    title={AdapterHub: A Framework for Adapting Transformers},
    author={Pfeiffer, Jonas and
            R{\"u}ckl{\'e}, Andreas and
            Poth, Clifton and
            Kamath, Aishwarya and
            Vuli{\'c}, Ivan and
            Ruder, Sebastian and
            Cho, Kyunghyun and
            Gurevych, Iryna},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    pages={46--54},
    year={2020}
}
```
