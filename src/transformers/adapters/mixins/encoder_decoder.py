from typing import Iterable, List, Optional, Tuple, Union

import torch.nn as nn

from transformers.adapters.composition import AdapterCompositionBlock, Fuse

from ..model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin


class EncoderDecoderModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the EncoderDecoderModel class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        if not isinstance(self.encoder, ModelAdaptersMixin) or not isinstance(self.decoder, ModelAdaptersMixin):
            return

        # Relay all invertible adapter calls to encoder
        self.invertible_adapters = self.encoder.base_model.invertible_adapters
        self.add_invertible_adapter = self.encoder.base_model.add_invertible_adapter
        self.get_invertible_adapter = self.encoder.base_model.get_invertible_adapter
        self.enable_invertible_adapters = self.encoder.base_model.enable_invertible_adapters
        self.invertible_adapters_forward = self.encoder.base_model.invertible_adapters_forward
        # Decoder should use invertible adapters of encoder
        self.decoder.base_model.invertible_adapters = self.encoder.base_model.invertible_adapters
        self.decoder.base_model.add_invertible_adapter = lambda *args: None
        self.decoder.base_model.get_invertible_adapter = self.encoder.base_model.get_invertible_adapter

        # Patched invertible adapters forward for decoder:
        # In the decoder, only the reverse pass in the LM head should be active.
        # Decoder inputs should not be forwarded through the invertible adapter again in its embeddings module.
        def decoder_invertible_adapters_forward(hidden_states, rev=False):
            if rev:
                return self.encoder.base_model.invertible_adapters_forward(hidden_states, rev=True)
            else:
                return hidden_states

        self.decoder.base_model.invertible_adapters_forward = decoder_invertible_adapters_forward

        super()._init_adapter_modules(add_prefix_tuning_pool=False)

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        self.encoder.add_adapter(adapter_name, config, overwrite_ok, set_active)

        if hasattr(config, "leave_out"):
            decoder_leave_out = [
                idx - self.encoder.config.num_hidden_layers
                for idx in config.leave_out
                if idx >= self.encoder.config.num_hidden_layers
            ]
            decoder_config = config.replace(leave_out=decoder_leave_out)
        else:
            decoder_config = config
        self.decoder.add_adapter(adapter_name, decoder_config, overwrite_ok, set_active)

    def delete_adapter(self, adapter_name: str):
        self.encoder.delete_adapter(adapter_name)
        self.decoder.delete_adapter(adapter_name)

    def add_fusion(self, adapter_names: Union[Fuse, list], adapter_fusion_config=None, override_kwargs=None):
        self.encoder.add_fusion(adapter_names, adapter_fusion_config, override_kwargs)

    def add_adapter_fusion(
        self,
        adapter_names: Union[Fuse, list, str],
        config=None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        self.encoder.add_adapter_fusion(adapter_names, config, overwrite_ok, set_active)
        self.decoder.add_adapter_fusion(adapter_names, config, overwrite_ok, set_active)

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        self.encoder.set_active_adapters(adapter_setup, skip_layers)
        self.decoder.set_active_adapters(adapter_setup, skip_layers)

    def _add_adapter_weights(self, adapter_name: str):
        self.encoder._add_adapter_weights(adapter_name)
        self.decoder._add_adapter_weights(adapter_name)

    def _count_parameters(self):
        num_param = sum(p.numel() for p in self.base_model.parameters())
        # prevent double counting shared parameters, because they are in the encoder and in the decoder
        num_param -= sum(p.numel() for p in self.decoder.shared_parameters.parameters())
        return num_param

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in self.encoder.iter_layers():
            yield i, layer
        for i, layer in self.decoder.iter_layers():
            yield i, layer
