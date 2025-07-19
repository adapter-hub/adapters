from adapters import AdapterModelInterface


CUSTOM_INTERFACES = {
    "modernbert": AdapterModelInterface(
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
    "gemma2": AdapterModelInterface(
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
    "gemma3_text": AdapterModelInterface(
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
        base_model="model",
    ),
    "qwen2": AdapterModelInterface(
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
        layer_pre_ffn="post_attention_layernorm",
        # Qwen applies layer norms before attention and MLP
        layer_ln_1=None,
        layer_ln_2=None,
    ),
    "qwen3": AdapterModelInterface(
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
        layer_pre_ffn="post_attention_layernorm",
        # Qwen applies layer norms before attention and MLP
        layer_ln_1=None,
        layer_ln_2=None,
    ),
    "phi": AdapterModelInterface(
        adapter_methods=["bottleneck", "lora", "reft", "invertible"],
        model_embeddings="embed_tokens",
        model_layers="layers",
        layer_self_attn="self_attn",
        layer_cross_attn=None,
        attn_k_proj="k_proj",
        attn_q_proj="q_proj",
        attn_v_proj="v_proj",
        attn_o_proj="dense",
        layer_intermediate_proj="mlp.fc1",
        layer_output_proj="mlp.fc2",
        layer_pre_self_attn="input_layernorm",
        layer_pre_cross_attn=None,
        # Phi applies one layer norm only before attention
        # these normalized hidden_states are then input to both the attention and the FFN
        # -> No layer_pre_ffn needed and also no normalization after the attention or FFN
        layer_pre_ffn=None,
        layer_ln_1=None,
        layer_ln_2=None,
    ),
}


def get_adapter_interface(model_name):
    if model_name in CUSTOM_INTERFACES:
        return CUSTOM_INTERFACES[model_name]
    else:
        return None
