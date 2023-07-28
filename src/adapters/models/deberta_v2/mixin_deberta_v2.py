from ...lora import Linear as LoRALinear
from ...prefix_tuning import PrefixTuningShim


class DebertaV2SelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, config):
        # Wrap layers for LoRA
        self.query_proj = LoRALinear.wrap(self.query_proj, "selfattn", config, attn_key="q")
        self.key_proj = LoRALinear.wrap(self.key_proj, "selfattn", config, attn_key="k")
        self.value_proj = LoRALinear.wrap(self.value_proj, "selfattn", config, attn_key="v")

        self.prefix_tuning = PrefixTuningShim(self.location_key + "_prefix" if self.location_key else None, config)
