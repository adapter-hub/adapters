from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer


class DebertaV2SelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.query_proj = LoRALinear.wrap(self.query_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.key_proj = LoRALinear.wrap(self.key_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.value_proj = LoRALinear.wrap(self.value_proj, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )
