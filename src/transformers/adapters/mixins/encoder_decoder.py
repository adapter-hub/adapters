from typing import Iterable, Tuple

import torch.nn as nn

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

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in self.encoder.iter_layers():
            yield i, layer
        for i, layer in self.decoder.iter_layers():
            yield i, layer
