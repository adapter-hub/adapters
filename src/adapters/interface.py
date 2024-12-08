from dataclasses import dataclass
from typing import List


class AdapterType:
    """
    Enum for the different adapter types.
    """

    bottleneck = "bottleneck"
    prefix_tuning = "prefix_tuning"
    lora = "lora"
    prompt_tuning = "prompt_tuning"
    reft = "reft"


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
        attn_v_proj (str): Name of the value projection layer in an attention layer
        layer_intermediate_proj (str): Name of the intermediate projection layer in a transformer layer.
        layer_output_proj (str): Name of the output projection layer in a transformer layer.
    """

    adapter_types: List[str]

    model_embeddings: str
    model_layers: str

    layer_self_attn: str
    layer_cross_attn: str
    attn_k_proj: str
    attn_q_proj: str
    attn_v_proj: str

    layer_intermediate_proj: str
    layer_output_proj: str
