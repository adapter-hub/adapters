# Integration with Hugging Face's Model Hub

```{eval-rst}
.. figure:: img/hfhub.svg
    :align: center
    :alt: Hugging Face Hub logo.
```

You can download adapters from and upload them to [Hugging Face's Model Hub](https://huggingface.co/models).
This document describes how to interact with the Model Hub when working with adapters.

## Downloading from the Hub

The Hugging Face Model Hub already provides a few pre-trained adapters available for download.
To search for available adapters, use the _Adapter Transformers_ library filter on the Model Hub website or use this link: [https://huggingface.co/models?filter=adapters](https://huggingface.co/models?filter=adapters).
Alternatively, all adapters on the Hugging Face Model Hub are also listed on [https://adapterhub.ml/explore](https://adapterhub.ml/explore) together with all adapters directly uploaded to AdapterHub.

After you have found an adapter you would like to use, loading it into a Transformer model is very similar to [loading adapters from AdapterHub](loading.md).
For example, for loading and activating the adapter [`AdapterHub/roberta-base-pf-sick`](https://huggingface.co/AdapterHub/roberta-base-pf-sick), write:
```python
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("roberta-base")
adapter_name = model.load_adapter("AdapterHub/roberta-base-pf-sick", source="hf")
model.active_adapters = adapter_name
```
Note that `source="hf"` is the only change from loading an adapter from AdapterHub.

## Uploading to the Hub

Hugging Face's Model Hub provides a convenient way for everyone to upload their pre-trained models and share them with the world.
Of course, this is also possible with adapters now!
In the following, we'll go through the fastest way of uploading an adapter directly via Python in the `adapters` library.
For more options and information, e.g. for managing models via the CLI and Git, refer to [HugginFace's documentation](https://huggingface.co/transformers/model_sharing.html).

1. **Prepare access credentials**: Before being able to push to the Hugging Face Model Hub for the first time, we have to store our access token in the cache.
    This can be done via the `transformers-cli` by running:
    ```
    transformers-cli login
    ```

2. **Push an adapter**: Next, we can proceed to upload our first adapter.
    Let's say we have a standard pre-trained Transformers model with an existing adapter named `awesome_adapter` (e.g. added via `model.add_adapter("awesome_adapter")` and [trained](training.md) afterwards).
    We can now push this adapter to the Model Hub using `model.push_adapter_to_hub()` like this:
    ```python
    model.push_adapter_to_hub(
        "my-awesome-adapter",
        "awesome_adapter",
        adapterhub_tag="sentiment/imdb",
        datasets_tag="imdb"
    )
    ```
    This will create a repository `my-awesome-adapter` under your username, generate a default adapter card as `README.md` and upload the adapter named `awesome_adapter` together with the adapter card to the new repository.
    `adapterhub_tag` and `datasets_tag` provide additional information for categorization.

    ```{eval-rst}
    .. important::
        All adapters uploaded to Hugging Face's Model Hub are automatically also listed on AdapterHub.ml. Thus, for better categorization, either ``adapterhub_tag`` or ``datasets_tag`` is required when uploading a new adapter to the Model Hub.

        - ``adapterhub_tag`` specifies the AdapterHub categorization of the adapter in the format ``<task>/<subtask>`` according to the tasks and subtasks shown on https://adapterhub.ml/explore. For more, see `Add a new task or subtask <https://docs.adapterhub.ml/contributing.html#add-a-new-task-or-subtask>`_.
        - ``datasets_tag`` specifies the dataset the adapter was trained on as an identifier from `Hugging Face Datasets <https://huggingface.co/datasets>`_.
    ```

Voilà! Your first adapter is on the Hugging Face Model Hub.
Anyone can now run:
```
model.load_adapter("<your_username>/my-awesome-adapter", source="hf")
```

To update your adapter, simply run `push_adapter_to_hub()` with the same repository name again. This will push a new commit to the existing repository.

You can find the full documentation of `push_adapter_to_hub()` [here](adapters.hub_mixin.PushAdapterToHubMixin.push_adapter_to_hub).
