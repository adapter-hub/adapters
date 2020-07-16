# Contributing to Adapter Hub

You can easily add your own pre-trained adapter modules or architectures to Adapter Hub via our [Hub GitHub repo](https://github.com/adapter-hub/hub). Please make sure to follow the steps below corresponding to the type of contribution you would like to make.

## Getting started

Before making any kind of contribution to _Adapter-Hub_, you will first need to setup your own fork of the _Hub_ repository to be able to make pull request later on:

1. Fork [the Hub repository](https://github.com/adapter-hub/hub) by clicking the 'Fork' button on the repository's page. This creates a clone of the repository under your GitHub user.

2. Clone your fork to your local file system:
```bash
git clone git@github.com:<YOUR_GITHUB_USER>/Hub.git
cd Hub
```

3. (optional) Set up the Python environment. This is useful to be able to perform some validation checks before submitting your contributions back to our repository:

```bash
pip install -r scripts/requirements.txt
```

As your fully set up now, you can proceed to working on your actual contribution now:

- [Add your pre-trained adapter](#add-your-pre-trained-adapter)
- [Add a new adapter architecture](#add-a-new-adapter-architecture)
- [Add a new task or subtask](#add-a-new-task-or-subtask)

## Add your pre-trained adapter

You can add your pre-trained adapter modules to the Hub so others can load them via `model.load_adapter()`.

_Note that we currently do not provide an option to host your module weights. Make sure you find an appropriate place to host them yourself!_

Let's go through the upload process step by step:

1. After the training of your adapter has finished, we first would want to save its weights to the local file system:
    ```python
    model.save_adapter("/path/to/save/folder", "your-adapter-name")
    ```

2. Zip the files in the created folder and upload the zip folder to your server space.

3. Now, we start the submission of your adapter to the Hub: After setting up your repository as described in the [Getting started section](#getting-started), create a subfolder for your user/ organization in the `adapters` folder.

    In your subfolder, create a YAML file describing your new adapter. You can start by copying the content of our [template YAML](https://github.com/adapter-hub/hub/blob/master/TEMPLATES/adapter.template.yaml). Only a few fields in the template are required for every submission but the more you fill, the better others can find your adapter. Especially make sure to set a download URL pointing to your uploaded weights folder.

    ```eval_rst
    .. note::
        The configuration of the adapter is specified by an identifier string in the YAML file. This string should refer to an adapter architecture available in the Hub. If you use a new or custom architecture, make sure to also `add an entry for your architecture <#add-a-new-adapter-architecture>`_ to the repo. 
    ```

4. (optional) After you completed filling the YAML adapter card, you can perform some validation checks to make sure your info card is correct (make sure to have the Python requirements set up as described in [Getting started](#getting-started)):

    ```
    python scripts/check.py -f adapters/<your_subfolder>/<your_adapter_card>
    ```

4. Almost finished: Now create [a pull request](https://github.com/Adapter-Hub/Hub/pulls) from your fork back to our repository.

    _We will perform some automatic checks on your PR to make sure the files you added are correct and the provided download links are valid. Keep an eye on the results of these checks!_

5. That's it! Your adapter will become available via our website as soon as your pull request is accepted! 🎉🚀


## Add a new adapter architecture

The `adapter-transformers` libraries has some common adapter configurations preincluded. However, if you want to add a new adapter using a different architecture, you can easily do this by adding the architecture configuration to the Hub repo:

1. After setting up your repository as described in the [Getting started section](#getting-started), create a new YAML file for your architecture in the `architectures` folder.

2. Fill in the full configuration dictionary of your architecture and some additional details. You can use [our template for architecture files](https://github.com/adapter-hub/hub/blob/master/TEMPLATES/adapter.template.yaml).

3. Create [a pull request](https://github.com/Adapter-Hub/Hub/pulls) from your fork back to our repository. 🚀


## Add a new task or subtask

Every adapter submitted to the Hub is identified by the task and the dataset (subtask) it was trained on. You're very encouraged to add additional information on the task and dataset of your adapter if they are not available yet. You can explore all currently available tasks at [https://adapterhub.ml/explore](https://adapterhub.ml/explore). To add a new task or subtask:

1. After setting up your repository as described in the [Getting started section](#getting-started), create a new YAML file for the task or subtask you would like to add in the `tasks` or `subtasks` folder.

2. Based on [our template for task files](https://github.com/adapter-hub/hub/blob/master/TEMPLATES/task.template.yaml) or [subtask files](https://github.com/adapter-hub/hub/blob/master/TEMPLATES/task.template.yaml), fill in some description and details on the task.

3. Create [a pull request](https://github.com/Adapter-Hub/Hub/pulls) from your fork back to our repository. 🚀
