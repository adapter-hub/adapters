from typing import Union

from ..composition import AdapterCompositionBlock
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


class EncoderDecoderModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the EncoderDecoderModel class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        self.encoder.train_adapter(adapter_setup)
        self.decoder.train_adapter(adapter_setup)

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.encoder.train_adapter_fusion(adapter_setup, unfreeze_adapters)
        self.decoder.train_adapter_fusion(adapter_setup, unfreeze_adapters)

    def _add_adapter(self, adapter_name):
        self.encoder.base_model._add_adapter(adapter_name)
        self.decoder.base_model._add_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.encoder.base_model._add_fusion_layer(adapter_names)
        self.decoder.base_model._add_fusion_layer(adapter_names)

    def _delete_adapter(self, adapter_name: str):
        self.encoder.base_model._delete_adapter(adapter_name)
        self.decoder.base_model._delete_adapter(adapter_name)

    def _delete_fusion_layer(self, adapter_names):
        self.encoder.base_model._delete_fusion_layer(adapter_names)
        self.decoder.base_model._delete_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        return self.encoder.get_fusion_regularization_loss() + self.decoder.get_fusion_regularization_loss()

    def get_adapter(self, name):
        return_adapters = self.encoder.get_adapter(name)
        for idx, items in self.decoder.get_adapter(name).items():
            return_adapters[len(return_adapters) + idx] = items

        return return_adapters
