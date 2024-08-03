from ...methods.lora import LoRAMergedLinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...utils import patch_forward
from ..bert.mixin_bert import BertModelAdaptersMixin


class DebertaSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.in_proj = LoRAMergedLinear.wrap(self.in_proj, "selfattn", model_config, adapters_config)

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )
        patch_forward(self)


class DebertaModelAdaptersMixin(BertModelAdaptersMixin):
    # Same as BERT, except that Deberta does not support the "lora_delta_w_svd" combine_strategy
    support_lora_delta_w_svd = False
