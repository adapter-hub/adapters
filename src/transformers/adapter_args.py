from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to adapter training.
    """

    train_adapter: bool = field(
        default=False, metadata={"help": "Train a text task adapter instead of the full model."}
    )
    load_task_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained task adapter to be loaded for further training."}
    )
    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language adapter to be loaded."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration."}
    )
