# Contributing Adapters to the Hub

```{eval-rst}
.. note::
    This document describes how to contribute adapters via the AdapterHub `Hub repository <https://github.com/adapter-hub/hub>`_. See `Integration with Hugging Face's Model Hub <huggingface_hub.html>`_ for uploading adapters via the Hugging Face Model Hub.
```

You can easily add your own pre-trained adapter modules or architectures to Adapter Hub via our [Hub GitHub repo](https://github.com/adapter-hub/hub). Please make sure to follow the steps below corresponding to the type of contribution you would like to make.

## Getting started

Before making any kind of contribution to _Adapter-Hub_, you will first need to set up your own fork of the _Hub_ repository to be able to open a pull request later on:

1. Fork [the Hub repository](https://github.com/adapter-hub/hub) by clicking the 'Fork' button on the repository's page. This creates a clone of the repository under your GitHub user.

2. Clone your fork to your local file system:
    ```bash
    git clone git@github.com:<YOUR_GITHUB_USER>/Hub.git
    cd Hub
    ```

3. Set up the Python environment. This includes the `adapter-hub-cli` which helps in preparing your adapters for the Hub.
    ```bash
    pip install -U ./scripts/.
    ```

As you're fully set up now, you can proceed on the specific steps if your contribution:

- [Contributing Adapters to the Hub](#contributing-adapters-to-the-hub)
  - [Getting started](#getting-started)
  - [Add your pre-trained adapter](#add-your-pre-trained-adapter)
  - [Add a new adapter architecture](#add-a-new-adapter-architecture)
  - [Add a new task or subtask](#add-a-new-task-or-subtask)

## Add your pre-trained adapter

You can add your pre-trained adapter modules to the Hub so others can load them via `model.load_adapter()`.

_Note that we currently do not provide an option to host your module weights. Make sure you find an appropriate place to host them yourself or consider uploading your adapter to the huggingface hub!_

Let's go through the upload process step by step:

1. After the training of your adapter has finished, we first would want to save its weights to the local file system:
    ```python
    model.save_adapter("/path/to/adapter/folder", "your-adapter-name")
    ```

2. Pack your adapter with the `adapter-hub-cli`. Start the CLI by giving it the path to your saved adapter:
    ```
    adapter-hub-cli pack /path/to/adapter/folder
    ```
    `adapter-hub-cli` will search for available adapters in the path you specify and interactively lead you through the packing process.

    ```{eval-rst}
    .. note::
        The configuration of the adapter is specified by an identifier string in the YAML file. This string should refer to an adapter architecture available in the Hub. If you use a new or custom architecture, make sure to also `add an entry for your architecture <#add-a-new-adapter-architecture>`_ to the repo. 
    ```

3. After step 2, a zipped adapter package and a corresponding YAML adapter card should have been created.
    - Upload the zip package to your server space and move the YAML file into a subfolder for your user/ organization in the `adapters` folder of the cloned Hub repository.
    - In the YAML adapter card, consider filling out some additional fields not filled out automatically, e.g. a description of your adapter is very useful!
    Especially make sure to set a download URL pointing to your uploaded zip package.

4. (optional) After you completed filling the YAML adapter card, you can perform some validation checks to make sure everything looks right:
    ```
    adapter-hub-cli check adapters/<your_subfolder>/<your_adapter_card>.yaml
    ```

5. Almost finished: Now create [a pull request](https://github.com/Adapter-Hub/Hub/pulls) from your fork back to our repository.

    _We will perform some automatic checks on your PR to make sure the files you added are correct and the provided download links are valid. Keep an eye on the results of these checks!_

6. That's it! Your adapter will become available via our website as soon as your pull request is accepted! 🎉🚀


## Add a new adapter architecture

The `adapters` libraries has some common adapter configurations preincluded. However, if you want to add a new adapter using a different architecture, you can easily do this by adding the architecture configuration to the Hub repo:

1. After setting up your repository as described in the [Getting started section](#getting-started), create a new YAML file for your architecture in the `architectures` folder.

2. Fill in the full configuration dictionary of your architecture and some additional details. You can use [our template for architecture files](https://github.com/adapter-hub/hub/blob/main/TEMPLATES/adapter.template.yaml).

3. Create [a pull request](https://github.com/Adapter-Hub/Hub/pulls) from your fork back to our repository. 🚀


## Add a new task or subtask

Every adapter submitted to the Hub is identified by the task and the dataset (subtask) it was trained on. You're very encouraged to add additional information on the task and dataset of your adapter if they are not available yet. You can explore all currently available tasks at [https://adapterhub.ml/explore](https://adapterhub.ml/explore). To add a new task or subtask:

1. After setting up your repository as described in the [Getting started section](#getting-started), create a new YAML file for the task or subtask you would like to add in the `tasks` or `subtasks` folder.

2. Based on [our template for task files](https://github.com/adapter-hub/hub/blob/main/TEMPLATES/task.template.yaml) or [subtask files](https://github.com/adapter-hub/hub/blob/main/TEMPLATES/task.template.yaml), fill in some description and details on the task.

3. Create [a pull request](https://github.com/Adapter-Hub/Hub/pulls) from your fork back to our repository. 🚀
