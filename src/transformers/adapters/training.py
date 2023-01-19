from dataclasses import dataclass, field
from typing import Optional

from .composition import Stack
from .configuration import AdapterConfigBase


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
        default="pfeiffer", metadata={"help": "Adapter configuration. Either a config string or a path to a file."}
    )


@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguments related to adapter training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )


def setup_adapter_training(model, adapter_args: AdapterArguments, adapter_name: str):
    """Setup model for adapter training based on given adapter arguments.

    Args:
        model (_type_): The model instance to be trained.
        adapter_args (AdapterArguments): The adapter arguments used for configuration.
        adapter_name (str): The name of the adapter to be added.
    """
    # Setup adapters
    if adapter_args.train_adapter:
        # resolve the adapter config
        adapter_config = AdapterConfigBase.load(adapter_args.adapter_config)
        # load a pre-trained from Hub if specified
        # note: this logic has changed in versions > 3.1.0: adapter is also loaded if it already exists
        if adapter_args.load_adapter:
            model.load_adapter(
                adapter_args.load_adapter,
                config=adapter_config,
                load_as=adapter_name,
            )
        # otherwise, if adapter does not exist, add it
        elif adapter_name not in model.config.adapters:
            model.add_adapter(adapter_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfigBase.load(adapter_args.lang_adapter_config)
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([adapter_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(Stack(lang_adapter_name, adapter_name))
        else:
            model.set_active_adapters(adapter_name)
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode.Use --train_adapter to enable adapter training"
            )
