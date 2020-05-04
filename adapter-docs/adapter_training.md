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

Training a task adapter module on a dataset only requires minor modifications from training the full model. Suppose we
have an existing script for training a transformer, here we will use the `run_glue.py` example script for training on
the GLUE dataset. The only main adaption we have to make is to add a new adapter module after the model is instantiated:

```python
# get actual model for derived models with heads
base_model = getattr(model, model.base_model_prefix, model)
# task adapter
base_model.set_adapter_config(AdapterType.text_task, args.adapter_config)
# load a pre-trained adapter for fine-tuning if specified
if args.load_task_adapter:
    base_model.load_task_adapter(args.load_task_adapter)
    tasks = base_model.config.text_task_adapters
# otherwise, add a new adapter
else:
    base_model.add_task_adapter(args.task_name)
    tasks = [args.task_name]
# enable adapter training
base_model.train_task_adapter()
```

```eval_rst
.. important::
    The most crucial step when training an adapter module is to freeze all weights in the model except for those of the
    adapter. In the previous snippet, this is achieved by calling the ``train_task_adapter()`` method which disables training
    of all weights outside the task adapter. In case you want to unfreeze all model weights later on, you can use
    ``freeze_model(False)``.
```

Besides this, we only have to make sure that the task we want to train is always passed as input to the `forward()` method
of the model (via the `adapter_tasks` parameter):

```python
inputs = {
    "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
    "adapter_tasks": tasks
}
outputs = model(**inputs)
```

You can find the full version of the modified training script for GLUE at
[run_glue.py](https://github.com/adapter-hub/adapter-transformers/blob/master/examples/run_glue.py)
in the *examples* folder of our repository.

To start adapter training on a GLUE task, you can run something similar to:

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=MNLI

python run_glue_tpu.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer
```

The important flag here is `--train_adapter` which switches from fine-tuning the full model to training an adapter
module for the given GLUE task.

## Train a Language Adapter

Training a language adapter is equally straightforward as training a task adapter. Similarly to the steps for task adapters
described above, we add a language adapter module to an existing model training script. Here, we modified the
[run_language_modeling.py](https://github.com/adapter-hub/adapter-transformers/blob/master/examples/run_language_modeling.py)
script.

Training a language adapter on BERT/RoBERTa then may look like the following:

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
    --mlm
    --train_language_adapter \
    --adapter_config pfeiffer
```
