# Examples

---
**NOTE for adapter-transformers: These are the example scripts from [HuggingFace Transformers](https://github.com/huggingface/transformers/tree/master/examples).
Some scripts have been modified to support adapter training (see table below).
For instructions on adapter training, please refer to the [Training section in the Adapter-Hub documentation](https://docs.adapterhub.ml/training).**

---

Version 2.9 of `transformers` introduced a new [`Trainer`](https://github.com/adapter-hub/adapter-transformers/blob/master/src/transformers/trainer.py) class for PyTorch, and its equivalent [`TFTrainer`](https://github.com/adapter-hub/adapter-transformers/blob/master/src/transformers/trainer_tf.py) for TF 2.
Running the examples requires PyTorch 1.3.1+ or TensorFlow 2.2+.

Here is the list of all our examples:
- **grouped by task** (all official examples work for multiple models)
- with information on whether support for training **Adapters** has been added to one of the example scripts (currently PyTorch only)
- with information on whether they are **built on top of `Trainer`/`TFTrainer`** (if not, they still work, they might just lack some features),
- whether or not they leverage the [🤗 Datasets](https://github.com/huggingface/datasets) library.


## The Big Table of Tasks

| Task | Example datasets | Adapter support (pytorch) | Trainer support | TFTrainer support | 🤗 Datasets
|---|---|:---:|:---:|:---:|:---:|
| [**`language-modeling`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/language-modeling)       | Raw text        | ✅ | ✅ | -  | ✅
| [**`text-classification`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/text-classification)   | GLUE, XNLI      | ✅ | ✅ | ✅ | ✅
| [**`token-classification`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/token-classification) | CoNLL NER       | ✅|  ✅ | ✅ | ✅
| [**`multiple-choice`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/multiple-choice)           | SWAG, RACE, ARC | ✅ | ✅ | ✅ | -
| [**`question-answering`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/question-answering)     | SQuAD           | ✅| ✅ | ✅ | -
| [**`text-generation`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/text-generation)           | -               | - | n/a | n/a | -
| [**`distillation`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/distillation)                 | All             | - | - | -  | -
| [**`summarization`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/seq2seq)                     | CNN/Daily Mail  | - | ✅  | - | -
| [**`translation`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/seq2seq)                       | WMT             | - | ✅  | - | -
| [**`bertology`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/bertology)                       | -               | - | - | - | -
| [**`adversarial`**](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/adversarial)                   | HANS            | - | ✅ | - | -


<br>

## Important note

**Important**

To make sure you can successfully run the latest versions of the example scripts, you have to **install the library from source** and install some example-specific requirements.
Execute the following steps in a new virtual environment:

```bash
git clone https://github.com/adapter-hub/adapter-transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

Alternatively, you can run the version of the examples as they were for your current version of Transformers via (for instance with v3.4.0):
```bash
git checkout tags/v3.4.0
```

## Running on TPUs

When using Tensorflow, TPUs are supported out of the box as a `tf.distribute.Strategy`.

When using PyTorch, we support TPUs thanks to `pytorch/xla`. For more context and information on how to setup your TPU environment refer to Google's documentation and to the
very detailed [pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

In this repo, we provide a very simple launcher script named [xla_spawn.py](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/xla_spawn.py) that lets you run our example scripts on multiple TPU cores without any boilerplate.
Just pass a `--num_cores` flag to this script, then your regular training script with its arguments (this is similar to the `torch.distributed.launch` helper for torch.distributed). 
Note that this approach does not work for examples that use `pytorch-lightning`.

For example for `run_glue`:

```bash
python examples/xla_spawn.py --num_cores 8 \
	examples/text-classification/run_glue.py \
	--model_name_or_path bert-base-cased \
	--task_name mnli \
	--data_dir ./data/glue_data/MNLI \
	--output_dir ./models/tpu \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--num_train_epochs 1 \
	--save_steps 20000
```

Feedback and more use cases and benchmarks involving TPUs are welcome, please share with the community.

## Logging & Experiment tracking

You can easily log and monitor your runs code. The following are currently supported:

* [TensorBoard](https://www.tensorflow.org/tensorboard)
* [Weights & Biases](https://docs.wandb.com/library/integrations/huggingface)
* [Comet ML](https://www.comet.ml/docs/python-sdk/huggingface/)

### Weights & Biases

To use Weights & Biases, install the wandb package with:

```bash
pip install wandb
```

Then log in the command line:

```bash
wandb login
```

If you are in Jupyter or Colab, you should login with:

```python
import wandb
wandb.login()
```

Whenever you use `Trainer` or `TFTrainer` classes, your losses, evaluation metrics, model topology and gradients (for `Trainer` only) will automatically be logged.

When using 🤗 Transformers with PyTorch Lightning, runs can be tracked through `WandbLogger`. Refer to related [documentation & examples](https://docs.wandb.com/library/integrations/lightning).

### Comet.ml

To use `comet_ml`, install the Python package with:

```bash
pip install comet_ml
```

or if in a Conda environment:

```bash
conda install -c comet_ml -c anaconda -c conda-forge comet_ml
```
