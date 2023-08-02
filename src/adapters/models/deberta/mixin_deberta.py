from ...lora import MergedLinear as LoRAMergedLinear
from ...prefix_tuning import PrefixTuningShim


class DebertaSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.in_proj = LoRAMergedLinear.wrap(self.in_proj, "selfattn", model_config, adapters_config)

        self.prefix_tuning = PrefixTuningShim(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )
