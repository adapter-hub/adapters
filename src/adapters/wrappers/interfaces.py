from adapters import AdapterModelInterface
from transformers import Gemma2PreTrainedModel, ModernBertPreTrainedModel


CUSTOM_INTERFACES = {
    ModernBertPreTrainedModel: AdapterModelInterface(
        adapter_methods=["bottleneck", "lora", "reft", "invertible"],  # not yet working for prompt tuning
        model_embeddings="embeddings",
        model_layers="layers",
        layer_self_attn="attn",
        layer_cross_attn=None,
        attn_qkv_proj="Wqkv",
        attn_o_proj="Wo",
        layer_intermediate_proj="mlp.Wi",
        layer_output_proj="mlp.Wo",
        layer_pre_self_attn="attn",
        layer_pre_cross_attn=None,
        layer_pre_ffn="mlp",
        layer_ln_1="mlp_norm",
        layer_ln_2=None,  # ModernBERT has no layer norm after the attention layer
    ),
    Gemma2PreTrainedModel: AdapterModelInterface(
        adapter_methods=["bottleneck", "lora", "reft", "invertible"],
        model_embeddings="embed_tokens",
        model_layers="layers",
        layer_self_attn="self_attn",
        layer_cross_attn=None,
        attn_k_proj="k_proj",
        attn_q_proj="q_proj",
        attn_v_proj="v_proj",
        attn_o_proj="o_proj",
        layer_intermediate_proj="mlp.up_proj",
        layer_output_proj="mlp.down_proj",
        layer_pre_self_attn="input_layernorm",
        layer_pre_cross_attn=None,
        layer_pre_ffn="pre_feedforward_layernorm",
        layer_ln_1="post_attention_layernorm",
        layer_ln_2="post_feedforward_layernorm",
    ),
}


def get_adapter_interface(model):
    # If it is a sub type of a known model, return the corresponding interface
    for model_type, interface in CUSTOM_INTERFACES.items():
        if isinstance(model, model_type):
            return interface
    return None
