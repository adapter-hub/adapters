# Prediction Heads

This section gives an overview how different prediction heads can be used together with adapter modules and how pretrained adapters can be distributed side-by-side with matching prediction heads in Adapter Hub.
We will take a look at our own implementation (e.g. `BertModelWithHeads`) as well as models with heads provided by Huggingface (e.g. `BertForSequenceClassification`).

## Models with flexible heads

To allow for prediction heads to be configured in a flexible way on top of a pretrained language model, `adapter-transformers` provides a new line of model classes.
These classes follow the naming schema `<model_class>WithHeads` and are available for all model classes supporting adapters. Let's see how they work:

First, we load pretrained model from Huggingface:
```python
model = BertModelWithHeads.from_pretrained("bert-base-uncased")
```

Although we use the class `BertModelWithHeads`, this model doesn't have any heads yet. We add a new one in the next step:
```python
model.add_classification_head("mrpc", num_labels=2)
```
The line above adds a binary sequence classification head on top of our model.
As this head is named, we could add multiple other heads with different names to the same model.
This is especially useful if used together with matching adapter modules.
For more about the different head types and the configuration options, refer to [the class reference](classes/bert_mixins.md#transformers.adapter_bert.BertModelHeadsMixin).

Now, of course, we would like to train our classification head together with an adapter, so let's add one:
```python
model.add_adapter("mrpc", config="pfeiffer")
model.set_active_adapters("mrpc")
```

Since we gave the task adapter the same name as our head, we can easily identify them as belonging together.
The call to `set_active_adapters()` in the second line tells our model to use the adapter - head configuration we specified by default in a forward pass.
At this point, we can start to [train our setup](training.md).

```eval_rst
.. note::
    The ``set_active_adapters()`` will search for an adapter module and a prediction head with the given name to be activated.
    If this method is not used, you can still activate a specific adapter module or prediction head by providing the `adapter_names` or `head` parameter in the forward call.
```

After training has completed, we can save our whole setup (adapter module _and_ prediction head), with a single call:
```python
model.save_adapter("/path/to/dir", "mrpc", with_head=True)
```

Now, we just have to [share our work with the world](contributing.md#add-your-pre-trained-adapter).
After we published our adapter together with its head in the Hub, anyone else can load both adapter and head by using the same model class.

Alternatively, we can also save and load the prediction head separately from an adapter module:

```python
# save
model.save_head("/path/to/dir", "mrpc")
# load
model.load_head("/path/to/dir")
```

## Huggingface heads

The `transformers` library provides strongly typed model classes with heads for various different tasks (e.g. `RobertaForSequenceClassification`, `AutoModelForMultipleChoice` ...).
If an adapter module is trained with one these out-of-the-box classes, it is encouraged to also distribute the prediction head weights together with the adapter weights.
Therefore, we can also easily save the prediction head weights for these models together with an adapter:

```python
model.save_adapter("/path/to/dir", "mrpc", with_head=True)
```

In the next step, we can provide both the adapter weights and the head weights to the Hub.
If someone else then downloads our pretrained adapter, the resolving method will check if our prediction head matches the class of his model.
In case the classes match, our prediction head weights will be automatically loaded too.
