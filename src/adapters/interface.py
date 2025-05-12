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

    Attributes:
        bottleneck: Adapter methods using bottleneck layers.
        prefix_tuning: Adapters methods based on Prefix Tuning. Note that this is currently unsupported via AdapterModelInterface.
        lora: Adapter methods based on low-rank adaptation.
        prompt_tuning: Adapter methods based on Prompt Tuning.
        reft: Adapters methods based on Representation Fine-Tuning.
        invertible: Adapter methods using invertible modules.
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
            for sub_config in config.configs:
                methods.extend(AdapterMethod.get_from_config(sub_config))
        else:
            methods.append(config.architecture)
        return methods


@dataclass
class AdapterModelInterface:
    """
    Defines the main interface for integrating adapter methods into a model class.
    This interface translates generic accessor names to model-specific attribute names.

    Args:
        adapter_methods (List[str]): List of adapter types that are supported by the model. Subset of this list: ["bottleneck", "lora", "reft", "prompt_tuning", "invertible"]
        model_embeddings (str): Name of the model's embedding layer.
        model_layers (str): Name of the model's layer list.
        layer_self_attn (str): Name of the self-attention layer in a transformer layer.
        layer_cross_attn (str): Name of the cross-attention layer in a transformer layer.
        attn_o_proj (str): Name of the output projection layer in an attention layer.
        layer_intermediate_proj (str): Name of the intermediate projection layer in a transformer layer.
        layer_output_proj (str): Name of the output projection layer in a transformer layer.

        # Either the following three attributes must be specified:
        attn_k_proj (Optional[str]): Name of the key projection layer in an attention layer.
        attn_q_proj (Optional[str]): Name of the query projection layer in an attention layer.
        attn_v_proj (Optional[str]): Name of the value projection layer in an attention layer.

        # Or this single attribute must be specified (but not both sets):
        attn_qkv_proj (Optional[str]): Name of the combined query-key-value projection layer (for models like GPT-2 or ModernBERT where QKV are in one tensor).

        # Optional attributes for extended bottleneck adapter support:
        layer_pre_self_attn (Optional[str]): Hook point directly before the self attention layer. Used for extended bottleneck adapter support.
        layer_pre_cross_attn (Optional[str]): Hook point directly before the cross attention layer. Used for extended bottleneck adapter support.
        layer_pre_ffn (Optional[str]): Hook point directly before the feed forward layer. Used for extended bottleneck adapter support.
        layer_ln_1 (Optional[str]): Layer norm *after* the self-attention layer. Used for extended bottleneck adapter support.
        layer_ln_2 (Optional[str]): Layer norm *after* the feed forward layer. Used for extended bottleneck adapter support.

        base_model (Optional[str]): Name of the base transformers model holding the layer modules. By default, this uses the model class' base_model_prefix attribute.

    Note:
        You must specify either all three of the individual projection layers (attn_k_proj, attn_q_proj, attn_v_proj) OR the combined projection layer (attn_qkv_proj).
    """

    adapter_methods: List[str]

    model_embeddings: str
    model_layers: str

    layer_self_attn: str
    layer_cross_attn: str

    attn_o_proj: Optional[str]

    layer_intermediate_proj: str
    layer_output_proj: str

    ###
    # Either all of these (this is the default and best working implementation):
    attn_k_proj: Optional[str] = None
    attn_q_proj: Optional[str] = None
    attn_v_proj: Optional[str] = None

    # Or this (for when query, key and value are stored in the same tensor as in GPT2 or ModernBERT):
    attn_qkv_proj: Optional[str] = None
    ###

    # Optional attributes for extended bottleneck adapter support
    layer_pre_self_attn: Optional[str] = None
    layer_pre_cross_attn: Optional[str] = None
    layer_pre_ffn: Optional[str] = None
    layer_ln_1: Optional[str] = None
    layer_ln_2: Optional[str] = None

    base_model: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        """Validate projection attributes after initialization."""

        has_separate_projections = (
            self.attn_k_proj is not None and self.attn_q_proj is not None and self.attn_v_proj is not None
        )
        has_combined_projection = self.attn_qkv_proj is not None

        if not has_separate_projections and not has_combined_projection:
            raise ValueError(
                "Must specify either individual projections (k,q,v) layers or combined qkv projection layer. You currently are neither specifying attn_qkv_proj nor attn_k_proj, attn_q_proj and attn_v_proj."
            )

        if has_separate_projections and has_combined_projection:
            raise ValueError(
                "Cannot specify both individual projections (k,q,v) and combined qkv projection. You specified attn_qkv_proj as well as attn_k_proj, attn_q_proj and attn_v_proj which makes no sense."
            )

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
