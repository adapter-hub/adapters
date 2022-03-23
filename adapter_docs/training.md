# Adapter Training

This section describes some examples on training different types of adapter modules in Transformer models.
The presented training scripts are only slightly modified from the original [examples by Huggingface](https://huggingface.co/transformers/examples.html).
To run the scripts, make sure you have the latest version of the repository and have installed some additional requirements:

```
git clone https://github.com/adapter-hub/adapter-transformers
cd transformers
pip install .
pip install -r ./examples/<your_examples_folder>/requirements.txt
```

## Train a Task Adapter

Training a task adapter module on a dataset only requires minor modifications from training the full model.
Suppose we have an existing script for training a Transformer model, here we will use HuggingFace's [run_glue.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/text-classification/run_glue.py) example script for training on the GLUE dataset.

In our example, we replaced the built-in `AutoModelForSequenceClassification` class with the `AutoAdapterModel` class introduced by `adapter-transformers` (learn more about prediction heads [here](prediction_heads.md)).
Therefore, the model instantiation changed to:

```python
model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
)
model.add_classification_head(data_args.task_name, num_labels=num_labels)
```

Compared to fine-tuning the full model, there is only one significant adaptation we have to make: adding a new adapter module and activating it.

```python
# task adapter - only add if not existing
if task_name not in model.config.adapters:
    # resolve the adapter config
    adapter_config = AdapterConfig.load(
        adapter_args.adapter_config,
        non_linearity=adapter_args.adapter_non_linearity,
        reduction_factor=adapter_args.adapter_reduction_factor,
    )
    # add a new adapter
    model.add_adapter(
        task_name,
        config=adapter_config
    )
# Enable adapter training
model.train_adapter(task_name)
```

```{eval-rst}
.. important::
    The most crucial step when training an adapter module is to freeze all weights in the model except for those of the
    adapter. In the previous snippet, this is achieved by calling the ``train_adapter()`` method which disables training
    of all weights outside the task adapter. In case you want to unfreeze all model weights later on, you can use
    ``freeze_model(False)``.
```

Besides this, we only have to make sure that the task adapter and prediction head are activated so that they are used in every forward pass. To specify the adapter modules to use, we can use the `model.set_active_adapters()` 
method and pass the adapter setup. If you only use a single adapter, you can simply pass the name of the adapter. For more information
on complex setups checkout the [Composition Blocks](https://docs.adapterhub.ml/adapter_composition.html).

```python
model.set_active_adapters(task_name)
```

The rest of the training procedure does not require any further changes in code.

You can find the full version of the modified training script for GLUE at [run_glue.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/text-classification/run_glue.py) in the `examples` folder of our repository.
We also adapted [various other example scripts](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples) (e.g. `run_glue.py`, `run_multiple_choice.py`, `run_squad.py`, ...) to support adapter training.

To start adapter training on a GLUE task, you can run something similar to:

```
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
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

```{eval-rst}
.. tip::
    Adapter weights are usually initialized randomly. That is why we require a higher learning rate. We have found that a default adapter learning rate of ``1e-4`` works well for most settings.
```

```{eval-rst}
.. tip::
    Depending on your data set size you might also need to train longer than usual. To avoid overfitting you can evaluating the adapters after each epoch on the development set and only save the best model.
```

## Train a Language Adapter

Training a language adapter is equally straightforward as training a task adapter. Similarly to the steps for task adapters
described above, we add a language adapter module to an existing model training script. Here, we modified HuggingFace's [run_mlm.py](https://github.com/Adapter-Hub/adapter-transformers/blob/v2/examples/language-modeling/run_mlm.py) script for masked language modeling with BERT-based models.

Training a language adapter on BERT using this script may look like the following:

```bash
export TRAIN_FILE=/path/to/dataset/train
export VALIDATION_FILE=/path/to/dataset/validation

python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir /tmp/test-mlm \
    --train_adapter \
    --adapter_config "pfeiffer+inv"
```

## Train AdapterFusion

We provide an example for training _AdapterFusion_ ([Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00247)) on the GLUE dataset: [run_fusion_glue.py](https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/adapterfusion/run_fusion_glue.py). 
You can adapt this script to train AdapterFusion with different pre-trained adapters on your own dataset.

```{eval-rst}
.. important::
    AdapterFusion on a target task is trained in a second training stage, after independently training adapters on individual tasks.
    When setting up a fusion architecture on your model, make sure to load the pre-trained adapter modules to be fused using ``model.load_adapter()`` before adding a fusion layer.
    For more on AdapterFusion, also refer to `Pfeiffer et al., 2020 <https://arxiv.org/pdf/2005.00247>`_.
```

To start fusion training on SST-2 as target task, you can run something like the following:

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2

python run_fusion_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir
```


## AdapterTrainer
Similar to the `Trainer` class provided by huggingface, adapter-transformers provides an `AdapterTrainer` class. This class is only
intended for training adapters. The `Trainer` class should still be used to fully fine-tune models. To train adapters with the `AdapterTrainer`
class, simply initialize it the same way you would initialize the `Trainer` class e.g.: 

```python
model.add_adapter(task_name)
model.train_adapter(task_name)

trainings_args =  TrainingsArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
)

trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
```
```{eval-rst}
.. tip::
    When you migrate from the previous versions, which use the Trainer class for adapter training and fully fine-tuning, note that the 
    specialized AdapterTrainer class does not have the parameters `do_save_full_model`, `do_save_adapters` and `do_save_adapter_fusion`.
```
