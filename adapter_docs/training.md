# Adapter Training

This section describes some examples on training different types of adapter modules in transformer models.
The presented training scripts are only slightly modified from the original [examples by Huggingface](https://huggingface.co/transformers/examples.html).
To run the scripts, make sure you have the latest version of the repository and have installed some additional requirements:

```
git clone https://github.com/adapter-hub/adapter-transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

## Train a Task Adapter

Training a task adapter module on a dataset only requires minor modifications from training the full model. Suppose we have an existing script for training a transformer, here we will use the `run_glue.py` example script for training on the GLUE dataset.

In our example, we replaced the built-in `AutoModelForSequenceClassification` class with the `AutoModelWithHeads` class introduced by `adapter-transformers` (learn more about prediction heads [here](prediction_heads.md)). Therefore, the model instantiation changed to:

```python
model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        config=config,
)
model.add_classification_head(data_args.task_name, num_labels=num_labels)
```

The only main adaption we have to make now is to add a new adapter module:

```python
# task adapter - only add if not existing
if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
    # add a new adapter
    model.add_adapter(
        task_name,
        AdapterType.text_task
        config=adapter_args.adapter_config
    )
# enable adapter training
model.train_adapter([task_name])
```

```eval_rst
.. important::
    The most crucial step when training an adapter module is to freeze all weights in the model except for those of the
    adapter. In the previous snippet, this is achieved by calling the ``train_adapter()`` method which disables training
    of all weights outside the task adapter. In case you want to unfreeze all model weights later on, you can use
    ``freeze_model(False)``.
```

Besides this, we only have to make sure that the task adapter and preddiction head are activated so that they are used in every forward pass:

```python
model.set_active_adapters(data_args.task_name)
```

The rest of the training procedure does not require any further changes in code.

You can find the full version of the modified training script for GLUE at [run_glue_wh.py](https://github.com/adapter-hub/adapter-transformers/blob/master/examples/run_glue_wh.py) in the *examples* folder of our repository.
We also adapted various other example scripts (e.g. `run_glue.py`, `run_multiple_choice.py`, `run_squad.py`, ...) to support adapter training.

To start adapter training on a GLUE task, you can run something similar to:

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=MNLI

python run_glue_wh.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer
```

The important flag here is `--train_adapter` which switches from fine-tuning the full model to training an adapter module for the given GLUE task.

## Train a Language Adapter

Training a language adapter is equally straightforward as training a task adapter. Similarly to the steps for task adapters
described above, we add a language adapter module to an existing model training script. Here, we modified the
[run_language_modeling.py](https://github.com/adapter-hub/adapter-transformers/blob/master/examples/run_language_modeling.py)
script by adding the following code:

```python
# language adapter - only add if not existing
if language not in model.config.adapters.adapter_list(AdapterType.text_lang):
    model.set_adapter_config(AdapterType.text_lang, adapter_args.adapter_config)
    model.add_adapter(language, AdapterType.text_lang)
# enable adapter training
model.train_adapter([language])
```

Training a language adapter on BERT then may look like the following:

```
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    -- language en \
    --train_adapter \
    --adapter_config pfeiffer
```
