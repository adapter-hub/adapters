# Prediction Heads

This section gives an overview how different prediction heads can be used together with adapter modules and how pre-trained adapters can be distributed side-by-side with matching prediction heads in AdapterHub.
We will take a look at our own new **model classes with flexible heads** (e.g. `BertModelWithHeads`) as well as **models with static heads** provided out-of-the-box by HuggingFace (e.g. `BertForSequenceClassification`).

```eval_rst
.. tip::
    We recommend to use the `model classes with flexible heads <#models-with-flexible-heads>`_ whenever possible.
    They have been created specifically for working with adapters and provide more flexibility.
```

```eval_rst
.. important::
    Although the two prediction head implementations serve the same use case, their weights are *not* directly compatible, i.e. you cannot load a head created with ``AutoModelWithHeds`` into a model of type ``AutoModelForSequenceClassification``.
    There is however an `automatic conversion to model classes with flexible heads <#automatic-conversion>`_.
```

## Models with flexible heads

To allow for prediction heads to be configured in a flexible way on top of a pre-trained language model, `adapter-transformers` provides a new line of model classes.
These classes follow the naming schema `<model_class>WithHeads` and are available for all model classes supporting adapters. Let's see how they work:

First, we load pre-trained model from HuggingFace:
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
For more about the different head types and the configuration options, refer to the class references of the respective model classes, e.g. [BertModelWithHeads](classes/models/bert.html#transformers.BertModelWithHeads).

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

Now, we just have to [share our work with the world](contributing.html#add-your-pre-trained-adapter).
After we published our adapter together with its head in the Hub, anyone else can load both adapter and head by using the same model class.

Alternatively, we can also save and load the prediction head separately from an adapter module:

```python
# save
model.save_head("/path/to/dir", "mrpc")
# load
model.load_head("/path/to/dir")
```

Lastly, it's also possible to delete an added head again:

```python
model.delete_head("mrpc")
```

## Model with static heads (HuggingFace heads)

The `transformers` library provides strongly typed model classes with heads for various different tasks (e.g. `RobertaForSequenceClassification`, `AutoModelForMultipleChoice` ...).
If an adapter module is trained with one these out-of-the-box classes, it is encouraged to also distribute the prediction head weights together with the adapter weights.
Therefore, we can also easily save the prediction head weights for these models together with an adapter:

```python
model.save_adapter("/path/to/dir", "mrpc", with_head=True)
```

In the next step, we can provide both the adapter weights and the head weights to the Hub.
If someone else then downloads our pre-trained adapter, the resolving method will check if our prediction head matches the class of his model.
In case the classes match, our prediction head weights will be automatically loaded too.

## Automatic conversion 

Beginning with v2.1 of `adapter-transformers`, it is possible to load static heads, e.g. created with `AutoModelForSequenceClassification`, into model classes with flexible heads, e.g. `AutoModelWithHeads`.
The conversion of weights happens automatically during the call of `load_adapter()`, so no additional steps are needed:
```python
static_head_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
static_head_model.add_adapter("test")
static_head_model.save_adapter(temp_dir, "test")

flex_head_model = AutoModelWithHeads.from_pretrained("bert-base-uncased")
flex_head_model.load_adapter(temp_dir)

assert "test" in flex_head_model.config.adapters
assert "test" in flex_head_model.heads
```

Note that a conversion in the opposite direction is not supported.
