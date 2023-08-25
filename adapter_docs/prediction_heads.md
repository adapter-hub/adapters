# Prediction Heads

This section gives an overview how different prediction heads can be used together with adapter modules and how pre-trained adapters can be distributed side-by-side with matching prediction heads in AdapterHub.
We will take a look at the `AdapterModel` classes (e.g. `BertAdapterModel`) introduced by adapter-transformers, which provide **flexible** support for prediction heads, as well as models with **static** heads provided out-of-the-box by HuggingFace Transformers (e.g. `BertForSequenceClassification`).

```{eval-rst}
.. tip::
    We recommend to use the `AdapterModel classes <#adaptermodel-classes>`_ whenever possible. 
    They have been created specifically for working with adapters and provide more flexibility.
```

## AdapterModel classes

The AdapterModel classes provided by `adapter-transformers` allow a flexible configuration of prediction heads on top of a pre-trained language model.

First, we load pre-trained model from the HuggingFace Hub via the [`AutoAdapterModel`](transformers.adapters.AutoAdapterModel) class:
```python
model = AutoAdapterModel.from_pretrained("bert-base-uncased")
```

By default, this model doesn't have any heads yet. We add a new one in the next step:
```python
model.add_classification_head("mrpc", num_labels=2)
```
The line above adds a binary sequence classification head on top of our model.
As this head is named, we could add multiple other heads with different names to the same model.
This is especially useful if used together with matching adapter modules.
To learn more about the different head types and the configuration options, please refer to the class references of the respective model classes, e.g. [`BertAdapterModel`](transformers.adapters.BertAdapterModel).

Now, of course, we would like to train our classification head together with an adapter, so let's add one:
```python
model.add_adapter("mrpc", config="pfeiffer")
model.set_active_adapters("mrpc")
```

Since we gave the task adapter the same name as our head, we can easily identify them as belonging together.
The call to `set_active_adapters()` in the second line tells our model to use the adapter - head configuration we specified by default in a forward pass.
At this point, we can start to [train our setup](training.md).

```{eval-rst}
.. note::
    The ``set_active_adapters()`` will search for an adapter and a prediction head with the given name to be activated.
    Alternatively, prediction heads can also be activated explicitly (i.e. without adapter modules).
    These three options are possible (in order of priority when multiple are specified):

    1. If ``head`` is passed to the forward call, the head with the given name is used.
    2. If the forward call is executed within an ``AdapterSetup`` context, the head configuration is read from the context.
    3. If the ``active_head`` property is set, the head configuration is read from there.
```

After training has completed, we can save our whole setup (adapter module _and_ prediction head), with a single call:
```python
model.save_adapter("/path/to/dir", "mrpc", with_head=True)
```

Now, you just have to [share your work with the world](./hub_contributing.md#add-your-pre-trained-adapter).
After you published our adapter together with its head in the Hub, anyone else can load both adapter and head by using the same model class.

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

## Model classes with static heads (HuggingFace Transformers)

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

```{eval-rst}
.. important::
    Although the two prediction head implementations serve the same use case, their weights are *not* directly compatible, i.e. you cannot load a head created with ``AutoAdapterModel`` into a model of type ``AutoModelForSequenceClassification``.
    There is however an automatic conversion to model classes with flexible heads.
```

Beginning with v2.1 of `adapter-transformers`, it is possible to load static heads, e.g. created with `AutoModelForSequenceClassification`, into model classes with flexible heads, e.g. `AutoAdapterModel`.
The conversion of weights happens automatically during the call of `load_adapter()`, so no additional steps are needed:
```python
static_head_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
static_head_model.add_adapter("test")
static_head_model.save_adapter(temp_dir, "test")

flex_head_model = AutoAdapterModel.from_pretrained("bert-base-uncased")
flex_head_model.load_adapter(temp_dir)

assert "test" in flex_head_model.config.adapters
assert "test" in flex_head_model.heads
```

Note that a conversion in the opposite direction is not supported.

## Custom Heads
If none of the available prediction heads fit your requirements, you can define and add a custom head.

First, we need to define the new head class. For that, the initialization and the forward pass need to be implemented.
The initialization of the head gets a reference to the model, the name of the head, and additionally defined kwargs. 
You can use the following template as a guideline.
```python 
class CustomHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        **kwargs,
    ):
        # innitialization of the custom head

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        # implementation of the forward pass
``` 


Next, we can register the new custom head and give the new head type a name. This only notifies
the model that there is a new head type. Then, we can add an instance of the new head to the model by
calling `add_custom_head` with the name of the new head type, the name of the head instance we are creating, and 
additional arguments required by the head.
```python
model.register_custom_head("my_custom_head", CustomHead)
model.add_custom_head(head_type="my_custom_head", head_name="custom_head", **kwargs)
```
After adding the custom head you can treat it like any other build-in head type.
