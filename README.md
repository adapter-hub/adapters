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

## `adapters` Library

This branch contains the development version of `adapters`, the next-generation library for parameter-efficient and modular transfer learning.

> **Note**: For the stable version of `adapter-transformers`, please switch to the [master branch of the repo](https://github.com/adapter-hub/adapter-transformers).

### Changes compared to `adapter-transformers`

- `adapters` is a standalone package, using `transformers` as an external dependency but not patching it directly
- All adapter-related classes now are imported via `adapters` namespace, e.g.:
    ```python
    from adapters import BertAdapterModel
    # ...
    ```
- Built-in HF model classes can be adapted for usage with adapters via a wrapper method, e.g.:
    ```python
    import adapters
    from transformers import BertModel

    model = BertModel.from_pretrained("bert-base-uncased")
    adapters.init(model)
    ```

Features not (yet) working:

- Loading model + adapter checkpoints using HF classes
- Using Transformers pipelines with adapters
- Using HF language modeling classes with invertible adapters

## Documentation
To read the documentation of _Adapters_, follow the steps in [docs/README.md](docs/README.md). Currently, the documentation is **not** yet available from https://docs.adapterhub.ml/.

---

<p align="center">
<img style="vertical-align:middle" src="https://raw.githubusercontent.com/Adapter-Hub/adapter-transformers/master/adapter_docs/logo.png" />
</p>
<h1 align="center">
<span><i>Adapters</i></span>
</h1>

<h3 align="center">
A Unified Library for Parameter-Efficient and Modular Transfer Learning
</h3>

![Tests](https://github.com/Adapter-Hub/adapter-transformers/workflows/Tests/badge.svg?branch=adapters)
[![GitHub](https://img.shields.io/github/license/adapter-hub/adapter-transformers.svg?color=blue)](https://github.com/adapter-hub/adapter-transformers/blob/adapters/LICENSE)
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
git clone https://github.com/adapter-hub/adapter-transformers.git
git checkout adapters
cd adapters
pip install .
```

## Getting Started

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
| Prefix Tuning | [Li and Liang (2021)](https://arxiv.org/pdf/2101.00190.pdf) | [Docs](https://docs.adapterhub.ml/overview.html#prefix-tuning) |
| Parallel adapters,<br> Mix-and-Match adapters | [He et al. (2021)](https://arxiv.org/pdf/2110.04366.pdf) | [Docs](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters) |
| Compacter | [Mahabadi et al. (2021)](https://arxiv.org/pdf/2106.04647.pdf) | [Docs](https://docs.adapterhub.ml/overview.html#compacter) |
| LoRA | [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf) | [Docs](https://docs.adapterhub.ml/overview.html#lora) |
| (IA)^3 | [Liu et al. (2022)](https://arxiv.org/pdf/2205.05638.pdf) | [Docs](https://docs.adapterhub.ml/overview.html#ia-3) |
| UniPELT | [Mao et al. (2022)](https://arxiv.org/pdf/2110.07577.pdf) | [Docs](https://docs.adapterhub.ml/overview.html#unipelt) |

## Supported Models

We currently support the PyTorch versions of all models listed on the **[Model Overview](https://docs.adapterhub.ml/model_overview.html) page** in our documentation.

## Citation

If you use this library for your work, please consider citing our paper [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779):

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
