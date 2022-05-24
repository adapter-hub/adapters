import logging
import os
import shutil
import stat
import tempfile
from typing import List, Optional, Union


logger = logging.getLogger(__name__)

DEFAULT_TEXT = "<!-- Add some description here -->"
# docstyle-ignore
ADAPTER_CARD_TEMPLATE = """
---
tags:
{tags}
---

# Adapter `{adapter_repo_name}` for {model_name}

An [adapter](https://adapterhub.ml) for the `{model_name}` model that was trained on the {dataset_name} dataset{head_info}.

This adapter was created for usage with the **[adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers)** library.

## Usage

First, install `adapter-transformers`:

```
pip install -U adapter-transformers
```
_Note: adapter-transformers is a fork of transformers that acts as a drop-in replacement with adapter support. [More](https://docs.adapterhub.ml/installation.html)_

Now, the adapter can be loaded and activated like this:

```python
from transformers import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("{model_name}")
adapter_name = model.load_adapter("{adapter_repo_name}", source="hf", set_active=True)
```

## Architecture & Training

{architecture_training}

## Evaluation results

{results}

## Citation

{citation}

"""


class PushAdapterToHubMixin:
    """Mixin providing support for uploading adapters to HuggingFace's Model Hub."""

    def _save_adapter_card(
        self,
        save_directory: str,
        adapter_name: str,
        adapter_repo_name: str,
        adapterhub_tag: Optional[str] = None,
        datasets_tag: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        license: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        **kwargs
    ):
        all_tags = {"adapter-transformers"}
        datasets = set()
        # Dataset/ Task info
        dataset_name = None
        if adapterhub_tag is None and datasets_tag is None:
            raise ValueError("Either adapterhub_tag or datasets_tag must be specified.")
        if datasets_tag is not None:
            dataset_name = f"[{datasets_tag}](https://huggingface.co/datasets/{datasets_tag}/)"
            datasets.add(datasets_tag)
        if adapterhub_tag is not None:
            # adapterhub_tag overwrites datasets_tag
            dataset_name = f"[{adapterhub_tag}](https://adapterhub.ml/explore/{adapterhub_tag}/)"
            all_tags.add(f"adapterhub:{adapterhub_tag}")

        all_tags.add(self.config.model_type)
        if tags is not None:
            all_tags = all_tags | set(tags)
        tag_string = "\n".join([f"- {tag}" for tag in all_tags])
        if datasets:
            tag_string += "\ndatasets:\n"
            tag_string += "\n".join([f"- {tag}" for tag in datasets])
        if language:
            tag_string += f"\nlanguage:\n- {language}"
        if license:
            tag_string += f'\nlicense: "{license}"'
        if metrics:
            tag_string += "\nmetrics:\n"
            tag_string += "\n".join([f"- {metric}" for metric in metrics])

        if hasattr(self, "heads") and adapter_name in self.heads:
            head_type = " ".join(self.heads[adapter_name].config["head_type"].split("_"))
            head_info = f" and includes a prediction head for {head_type}"
        else:
            head_info = ""

        adapter_card = ADAPTER_CARD_TEMPLATE.format(
            tags=tag_string,
            model_name=self.model_name,
            dataset_name=dataset_name,
            head_info=head_info,
            adapter_repo_name=adapter_repo_name,
            architecture_training=kwargs.pop("architecture_training", DEFAULT_TEXT),
            results=kwargs.pop("results", DEFAULT_TEXT),
            citation=kwargs.pop("citation", DEFAULT_TEXT),
        )

        logger.info('Saving adapter card for adapter "%s" to %s.', adapter_name, save_directory)
        with open(os.path.join(save_directory, "README.md"), "w") as f:
            f.write(adapter_card.strip())

    def push_adapter_to_hub(
        self,
        repo_name: str,
        adapter_name: str,
        organization: Optional[str] = None,
        adapterhub_tag: Optional[str] = None,
        datasets_tag: Optional[str] = None,
        local_path: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Union[bool, str] = True,
        overwrite_adapter_card: bool = False,
        adapter_card_kwargs: Optional[dict] = None,
    ):
        """Upload an adapter to HuggingFace's Model Hub.

        Args:
            repo_name (str): The name of the repository on the model hub to upload to.
            adapter_name (str): The name of the adapter to be uploaded.
            organization (str, optional): Organization in which to push the adapter
                (you must be a member of this organization). Defaults to None.
            adapterhub_tag (str, optional):
                Tag of the format `<task>/<subtask>` for categorization on https://adapterhub.ml/explore/. See
                https://docs.adapterhub.ml/contributing.html#add-a-new-task-or-subtask for more. If not specified,
                `datasets_tag` must be given in case a new adapter card is generated. Defaults to None.
            datasets_tag (str, optional): Dataset identifier from https://huggingface.co/datasets.
                If not specified, `adapterhub_tag` must be given in case a new adapter card is generated. Defaults to
                None.
            local_path (str, optional): Local path used as clone directory of the adapter repository.
                If not specified, will create a temporary directory. Defaults to None.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add config"`, :obj:`"add tokenizer"` or
                :obj:`"add model"` depending on the type of the class.
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Defaults to
                True.
            overwrite_adapter_card (bool, optional): Overwrite an existing adapter card with a newly generated one.
                If set to `False`, will only generate an adapter card, if none exists. Defaults to False.

        Returns:
            str: The url of the adapter repository on the model hub.
        """
        repo_url = self._get_repo_url_from_name(
            repo_name, organization=organization, private=private, use_auth_token=use_auth_token
        )
        if local_path is not None:
            repo_path = local_path
        else:
            repo_path = tempfile.mkdtemp()
        # Create repo or get retrieve an existing repo
        repo = self._create_or_get_repo(
            repo_path_or_name=repo_path,
            repo_url=repo_url,
            organization=organization,
            private=private,
            use_auth_token=use_auth_token,
        )
        # Save adapter and optionally create model card
        self.save_adapter(repo_path, adapter_name)
        if overwrite_adapter_card or not os.path.exists(os.path.join(repo_path, "README.md")):
            full_repo_name = "/".join(repo_url.split("/")[-2:])
            adapter_card_kwargs = adapter_card_kwargs or {}
            self._save_adapter_card(
                repo_path,
                adapter_name,
                full_repo_name,
                adapterhub_tag=adapterhub_tag,
                datasets_tag=datasets_tag,
                **adapter_card_kwargs,
            )
        # Commit and push
        logger.info('Pushing adapter "%s" to model hub at %s ...', adapter_name, repo_url)
        url = self._push_to_hub(repo, commit_message=commit_message)

        # Clean up if temp dir was used
        if local_path is None:

            def on_rm_error(func, path, exc_info):
                # path contains the path of the file that couldn't be removed
                # let's just assume that it's read-only and unlink it.
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.unlink(path)
                except Exception:
                    pass

            shutil.rmtree(repo_path, onerror=on_rm_error)

        return url
