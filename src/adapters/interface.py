from dataclasses import dataclass, field
from typing import List


class AdapterType:
    bottleneck = "bottleneck"
    prefix_tuning = "prefix_tuning"
    lora = "lora"
    prompt_tuning = "prompt_tuning"
    reft = "reft"


@dataclass
class AdapterModelInterface:
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
