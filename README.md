<p align="center">
<img style="vertical-align:middle" src="https://raw.githubusercontent.com/Adapter-Hub/adapter-transformers/master/adapter_docs/logo.png" />
</p>
<h1 align="center">
<span>adapter-transformers</span>
</h1>

<h3 align="center">
A friendly fork of HuggingFace's <i>Transformers</i>, adding Adapters to PyTorch language models
</h3>

![Tests](https://github.com/Adapter-Hub/adapter-transformers/workflows/Tests/badge.svg)
[![GitHub](https://img.shields.io/github/license/adapter-hub/adapter-transformers.svg?color=blue)](https://github.com/adapter-hub/adapter-transformers/blob/master/LICENSE)
![PyPI](https://img.shields.io/pypi/v/adapter-transformers)

`adapter-transformers` is an extension of [HuggingFace's Transformers](https://github.com/huggingface/transformers) library, integrating adapters into state-of-the-art language models by incorporating **[AdapterHub](https://adapterhub.ml)**, a central repository for pre-trained adapter modules.

This library can be used as a drop-in replacement for HuggingFace Transformers and regularly synchronizes new upstream changes.

## Quick tour

_adapter-transformers_ currently supports **Python 3.6+** and **PyTorch 1.1.0+**.
After [installing PyTorch](https://pytorch.org/get-started/locally/), you can install _adapter-transformers_ from PyPI ...

```
pip install -U adapter-transformers
```

... or from source by cloning the repository:

```
git clone https://github.com/adapter-hub/adapter-transformers.git
cd adapter-transformers
pip install .
```

## Getting Started

HuggingFace's great documentation on getting started with _Transformers_ can be found [here](https://huggingface.co/transformers/index.html). _adapter-transformers_ is fully compatible with _Transformers_.

To get started with adapters, refer to these locations:

- **[Colab notebook tutorials](https://github.com/Adapter-Hub/adapter-transformers/tree/master/notebooks)**, a series notebooks providing an introduction to all the main concepts of (adapter-)transformers and AdapterHub
- **https://docs.adapterhub.ml**, our documentation on training and using adapters with _adapter-transformers_
- **https://adapterhub.ml** to explore available pre-trained adapter modules and share your own adapters
- **[Examples folder](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples)** of this repository containing HuggingFace's example training scripts, many adapted for training adapters


## Citation

If you find this library useful, please cite our paper [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779):

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
