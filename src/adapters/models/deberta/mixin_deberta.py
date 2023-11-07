from ...methods.lora import LoRAMergedLinear
from ...methods.prefix_tuning import PrefixTuningLayer


class DebertaSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.in_proj = LoRAMergedLinear.wrap(self.in_proj, "selfattn", model_config, adapters_config)

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )
