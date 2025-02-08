import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from transformers.utils import cached_file

from . import __version__
from .utils import INTERFACE_CONFIG_NAME


class AdapterMethod:
    """
    Enum of all supported adapter method types.
    """

    bottleneck = "bottleneck"
    prefix_tuning = "prefix_tuning"
    lora = "lora"
    prompt_tuning = "prompt_tuning"
    reft = "reft"
    invertible = "invertible"

    @staticmethod
    def get_from_config(config) -> List[str]:
        """
        Get the adapter type from a given adapter config.

        Args:
            config: The adapter config.

        Returns:
            List[str]: The adapter type.
        """
        methods = []
        if getattr(config, "inv_adapter", False):
            methods.append(AdapterMethod.invertible)
        if config.architecture is None:
            methods.append(AdapterMethod.bottleneck)
        elif config.architecture == "union":
            methods.extend([AdapterMethod.get_from_config(sub_config) for sub_config in config.configs])
        else:
            methods.append(config.architecture)
        return methods


@dataclass
class AdapterModelInterface:
    """
    Defines the main interface for integrating adapter methods into a model class.
    This interface translates generic accessor names to model-specific attribute names.

    Args:
        adapter_types (List[str]): List of adapter types that are supported by the model.
        model_embeddings (str): Name of the model's embedding layer.
        model_layers (str): Name of the model's layer list.
        layer_self_attn (str): Name of the self-attention layer in a transformer layer.
        layer_cross_attn (str): Name of the cross-attention layer in a transformer layer.
        attn_k_proj (str): Name of the key projection layer in an attention layer.
        attn_q_proj (str): Name of the query projection layer in an attention layer.
        attn_v_proj (str): Name of the value projection layer in an attention layer.
        attn_o_proj (str): Name of the output projection layer in an attention layer.
        layer_intermediate_proj (str): Name of the intermediate projection layer in a transformer layer.
        layer_output_proj (str): Name of the output projection layer in a transformer layer.
        layer_pre_self_attn (Optional[str]): Hook point directly before the self attention layer. Used for extended bottleneck adapter support.
        layer_pre_cross_attn (Optional[str]): Hook point directly before the cross attention layer. Used for extended bottleneck adapter support.
        layer_pre_ffn (Optional[str]): Hook point directly before the feed forward layer. Used for extended bottleneck adapter support.
        layer_ln_1 (Optional[str]): Layer norm *after* the self-attention layer. Used for extended bottleneck adapter support.
        layer_ln_2 (Optional[str]): Layer norm *after* the feed forward layer. Used for extended bottleneck adapter support.
    """

    adapter_types: List[str]

    model_embeddings: str
    model_layers: str

    layer_self_attn: str
    layer_cross_attn: str
    attn_k_proj: str
    attn_q_proj: str
    attn_v_proj: str
    attn_o_proj: str

    layer_intermediate_proj: str
    layer_output_proj: str

    # Optional attributes for extended bottleneck adapter support
    layer_pre_self_attn: Optional[str] = None
    layer_pre_cross_attn: Optional[str] = None
    layer_pre_ffn: Optional[str] = None
    layer_ln_1: Optional[str] = None
    layer_ln_2: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def _save(self, save_directory, model_config):
        config_dict = {
            "model_type": model_config.model_type,
            "interface": self.to_dict(),
            "version": "adapters." + __version__,
        }
        save_path = os.path.join(save_directory, INTERFACE_CONFIG_NAME)
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

    @classmethod
    def _load(cls, path_or_repo_id: str, **kwargs):
        resolved_file = cached_file(path_or_repo_id, INTERFACE_CONFIG_NAME, **kwargs)
        with open(resolved_file, "r") as f:
            config_dict = json.load(f)
        return AdapterModelInterface(**config_dict["interface"])
