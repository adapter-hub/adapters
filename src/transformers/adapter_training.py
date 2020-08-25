from dataclasses import dataclass, field
from typing import Optional

from .adapter_bert import BertModelHeadsMixin
from .adapter_config import AdapterConfig, AdapterType


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to adapter training.
    """

    train_adapter: bool = field(default=False, metadata={"help": "Train an adapter instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[str] = field(
        default=None, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})


@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to adapter training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language adapter configuration."}
    )
    lang_adapter_reduction_factor: Optional[str] = field(
        default=None, metadata={"help": "Override the reduction factor of the language adapter configuration."}
    )


def setup_task_adapter_training(model, task_name: str, adapter_args: AdapterArguments):
    """Sets up task adapter training for a given pre-trained model.

    Args:
        model (PretrainedModel): The model for which to set up task adapter training.
        task_name (str): The name of the task to train.
        adapter_args (AdapterArguments): Adapter traininf arguments.
    """
    language = adapter_args.load_lang_adapter
    if adapter_args.train_adapter:
        # task adapter - only add if not existing
        if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
            # load a pre-trained adapter for fine-tuning if specified
            if adapter_args.load_task_adapter:
                model.load_adapter(
                    adapter_args.load_task_adapter,
                    AdapterType.text_task,
                    config=adapter_args.adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a new adapter
            else:
                model.add_adapter(task_name, AdapterType.text_task, config=adapter_args.adapter_config)
        # language adapter - only add if not existing
        if language and language not in model.config.adapters.adapter_list(AdapterType.text_lang):
            lconfig_string = adapter_args.lang_adapter_config or adapter_args.adapter_config
            # TODO support different non_linearity & reduction_factor
            lconfig = AdapterConfig.load(lconfig_string, non_linearity="gelu", reduction_factor=2)

            model.load_adapter(
                adapter_args.load_lang_adapter, AdapterType.text_lang, config=lconfig, load_as=adapter_args.language
            )
        # enable adapter training
        model.train_adapter([task_name])
    # set adapters as default if possible
    if isinstance(model, BertModelHeadsMixin):
        adapter_names = []
        if language:
            adapter_names.append([language])
        adapter_names.append([task_name])
        model.set_active_adapters(adapter_names)
