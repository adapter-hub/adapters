from typing import Iterable, Tuple

import torch.nn as nn

from transformers.utils import logging

from ...composition import adjust_tensors_for_parallel_
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


logger = logging.get_logger(__name__)


class XmodModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        # Delete original adapter modules
        for _, layer in self.iter_layers():
            del layer.output.adapter_modules

        # Register hook for post embedding forward
        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            # hook[1] is lang_ids tensor
            adjust_tensors_for_parallel_(input[0], input[2])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer

    def forward(self, *args, **kwargs):
        if "lang_ids" in kwargs and kwargs["lang_ids"] is not None:
            raise ValueError(
                "XmodModel with adapters does not support `lang_ids` as an argument. Use `set_active_adapters`"
                " instead."
            )
        else:
            kwargs["lang_ids"] = 1
        return super().forward(*args, **kwargs)

    # Override adapter-specific methods in original implementation

    def set_default_language(self, language: str):
        raise ValueError(
            "`set_default_language` is not implemented for models using `adapters`. Use `set_active_adapters` instead."
        )

    def freeze_embeddings_and_language_adapters(self):
        """
        Freeze the embeddings and language adapters of the model. Usually, this is applied before the model is
        fine-tuned on a downstream task.
        """
        # TODO: Replace this by a general method for `adapters`.
        logger.info("Freezing embeddings")
        for parameter in self.base_model.embeddings.parameters():
            parameter.requires_grad = False
        logger.info("Freezing adapters")
        for adapter_name in self.adapters_config:
            self.apply_to_adapter_layers(lambda i, layer: layer.freeze_adapter(adapter_name))
