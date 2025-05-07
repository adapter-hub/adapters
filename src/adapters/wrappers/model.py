import importlib
import os
from typing import Any, Optional, Type, Union

from torch import nn

from adapters.context import ForwardContext
from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.models.auto.configuration_auto import model_type_to_module_name

from ..configuration import ModelAdaptersConfig
from ..interface import AdapterModelInterface
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    ModelAdaptersMixin,
    ModelBaseAdaptersMixin,
    ModelUsingSubmodelsAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)
from ..models import MODEL_MIXIN_MAPPING
from ..utils import multigetattr, multihasattr
from .configuration import init_adapters_config
from .interfaces import get_adapter_interface


SPECIAL_MODEL_TYPE_TO_MODULE_NAME = {
    "clip_vision_model": "clip",
    "clip_text_model": "clip",
}


_INTERFACE_ERROR_TEMPLATE = "AdapterInterface: '{layer_name}' is set to '{layer_value}' but this value is not found in the {parent_name}. See https://docs.adapterhub.ml/plugin_interface.html for more information."


def get_module_name(model_type: str) -> str:
    if model_type in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[model_type]
    return model_type_to_module_name(model_type)


def replace_with_adapter_class(module: nn.Module, modules_with_adapters) -> None:
    # Check if module is a base model class
    if module.__class__.__name__ in MODEL_MIXIN_MAPPING:
        # Create new wrapper model class
        model_class = type(
            module.__class__.__name__,
            (MODEL_MIXIN_MAPPING[module.__class__.__name__], module.__class__),
            {},
        )
        module.__class__ = model_class
    elif module.__class__.__module__.startswith("transformers.models") or module.__class__.__module__.startswith(
        "adapters.wrappers.model"
    ):
        try:
            module_class = getattribute_from_module(
                modules_with_adapters,
                module.__class__.__name__ + "WithAdapters",
            )
            module.__class__ = module_class
        except ValueError:
            # Silently fail and keep original module class
            pass


def init(
    model: PreTrainedModel,
    adapters_config: Optional[ModelAdaptersConfig] = None,
    interface: Optional[AdapterModelInterface] = None,
) -> None:
    if isinstance(model, ModelAdaptersMixin):
        return model

    model_name = get_module_name(model.config.model_type)

    # If interface is None, have a look at our pre-supported interfaces
    if interface is None:
        interface = get_adapter_interface(model.config.model_type)

    if interface is not None:
        # Override the default base_model_prefix
        if base_model_prefix := interface.base_model:
            model.base_model_prefix = base_model_prefix
        base_model = model.base_model
        _validate_interface_values(base_model, interface)
        model_class_name = base_model.__class__.__name__
        model_class = type(
            model_class_name,
            (
                EmbeddingAdaptersMixin,
                ModelBaseAdaptersMixin,
                base_model.__class__,
            ),
            {},
        )
        base_model.__class__ = model_class
        base_model.adapter_interface = interface
        base_model.support_prompt_tuning = False  # HACK: will be set to true if init_prompt_tuning() is called
    else:
        # First, replace original module classes with their adapters counterparts
        try:
            modules_with_adapters = importlib.import_module(f".{model_name}.modeling_{model_name}", "adapters.models")
        except ImportError:
            raise ValueError(
                f"Model {model_name} not pre-supported by adapters. Please specify and pass `interface` explicitly."
            )
        submodules = list(model.modules())

        # Replace the base model class
        replace_with_adapter_class(submodules.pop(0), modules_with_adapters)

        # Check if the base model class derives from ModelUsingSubmodelsAdaptersMixin
        if isinstance(model, ModelUsingSubmodelsAdaptersMixin):
            # Before initializing the submodels, make sure that adapters_config is set for the whole model.
            # Otherwise, it would not be shared between the submodels.
            init_adapters_config(model, model.config, adapters_config)
            adapters_config = model.adapters_config
            model.init_submodels()
            submodules = []

        # Change the class of all child modules to their adapters class
        for module in submodules:
            replace_with_adapter_class(module, modules_with_adapters)

    # Next, check if model class itself is not replaced and has an adapter-supporting base class
    if not isinstance(model, ModelAdaptersMixin):
        if hasattr(model, "base_model_prefix") and hasattr(model, model.base_model_prefix):
            base_model = getattr(model, model.base_model_prefix)
            if isinstance(base_model, ModelAdaptersMixin):
                # HACK to preserve original forward method signature (e.g. for Trainer label names)
                temp_signature = ForwardContext.add_context_args_in_signature(model.forward.__func__)
                # Create new wrapper model class
                model_class_name = model.__class__.__name__
                model_class = type(
                    model_class_name,
                    (
                        EmbeddingAdaptersWrapperMixin,
                        ModelWithHeadsAdaptersMixin,
                        model.__class__,
                    ),
                    {},
                )
                model.__class__ = model_class
                model.forward.__func__.__signature__ = temp_signature

    # Finally, initialize adapters
    model.init_adapters(model.config, adapters_config)


def load_model(
    model_name_or_path: Optional[Union[str, os.PathLike]],
    model_class: Type[PreTrainedModel],
    interface: Optional[AdapterModelInterface] = None,
    *model_args: Any,
    **kwargs: Any,
) -> PreTrainedModel:
    """
    Loads a pretrained model with adapters from the given path or url.

    Parameters:
        model_name_or_path (`str` or `os.PathLike`, *optional*):
            Parameter identical to PreTrainedModel.from_pretrained
        model_class (`PreTrainedModel` or `AutoModel`):
            The model class to load (e.g. EncoderDecoderModel and EncoderDecoderAdapterModel both work)
        interface (`AdapterModelInterface`, *optional*):
            The custom adapter interface to use for the model, to be passed to the init() method.
            If not provided, init() will try to use one of the built-in model integrations.
        model_args (sequence of positional arguments, *optional*):
            All remaining positional arguments will be passed to the underlying model's `__init__` method.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
            `output_attentions=True`).
    Returns:
        `PreTrainedModel`: The model with adapters loaded from the given path or url.
    """

    old_init = model_class.__init__

    # try if we can find a interface file
    if interface is None:
        try:
            interface = AdapterModelInterface._load(model_name_or_path, **kwargs)
        except EnvironmentError:
            pass

    def new_init(self, config, *args, **kwargs):
        old_init(self, config, *args, **kwargs)
        init(self, interface=interface)

    # wrap model after it is initialized but before the weights are loaded
    new_model_class = type(model_class.__name__, (model_class,), {})
    new_model_class.__init__ = new_init
    model = new_model_class.from_pretrained(model_name_or_path, *model_args, **kwargs)

    return model


def _validate_interface_values(base_model: PreTrainedModel, interface: AdapterModelInterface) -> None:
    """
    Validates that all values specified in the interface exist in the model.

    Args:
        base_model: The base model to validate against
        interface: The adapter interface to validate

    Raises:
        ValueError: If any specified path is not found in the model
    """

    if not multihasattr(base_model, interface.model_embeddings):
        raise ValueError(
            _INTERFACE_ERROR_TEMPLATE.format(
                layer_name="model_embeddings",
                layer_value=interface.model_embeddings,
                parent_name="base_model",
            )
        )
    # All other values are layer specific => Get the first layer and check if all values are present
    layers = multigetattr(base_model, interface.model_layers)
    if not layers:
        raise ValueError(
            _INTERFACE_ERROR_TEMPLATE.format(
                layer_name="model_layers",
                layer_value=interface.model_layers,
                parent_name="base_model",
            )
        )

    if len(layers) == 0:
        raise ValueError(
            f"AdapterInterface: 'model_layers' is set to '{interface.model_layers}'. But accessing this value of the base_model returns an empty list. See https://docs.adapterhub.ml/plugin_interface.html for more information."
        )

    layer = layers[0]

    layer_attributes = [
        "layer_self_attn",
        "layer_cross_attn",
        "layer_intermediate_proj",
        "layer_output_proj",
        "layer_pre_self_attn",
        "layer_pre_cross_attn",
        "layer_pre_ffn",
        "layer_ln_1",
        "layer_ln_2",
    ]
    values_to_check = {
        name: getattr(interface, name) for name in layer_attributes if getattr(interface, name) is not None
    }

    for layer_name, layer_value in values_to_check.items():
        if not multihasattr(layer, layer_value):
            raise ValueError(
                _INTERFACE_ERROR_TEMPLATE.format(
                    layer_name=layer_name,
                    layer_value=layer_value,
                    parent_name="model layer",
                )
            )

    # Check attention-specific attributes if self-attention or cross-attention is defined
    attention_attributes = ["attn_o_proj"]

    if getattr(interface, "attn_q_proj") is not None:
        attention_attributes += ["attn_q_proj", "attn_k_proj", "attn_v_proj"]
    else:
        # If q,k,v are not specified on their own, they must be specified combined in attn_qkv_proj
        attention_attributes += ["attn_qkv_proj"]

    if interface.layer_self_attn is not None:
        self_attn_module = multigetattr(layer, interface.layer_self_attn)
        for attn_name in attention_attributes:
            attn_value = getattr(interface, attn_name)
            if not multihasattr(self_attn_module, attn_value):
                raise ValueError(
                    _INTERFACE_ERROR_TEMPLATE.format(
                        layer_name=attn_name,
                        layer_value=attn_value,
                        parent_name="self-attention layer",
                    )
                )

    if interface.layer_cross_attn is not None:
        cross_attn_module = multigetattr(layer, interface.layer_cross_attn)
        for attn_name in attention_attributes:
            attn_value = getattr(interface, attn_name)
            if not multihasattr(cross_attn_module, attn_value):
                raise ValueError(
                    _INTERFACE_ERROR_TEMPLATE.format(
                        layer_name=attn_name,
                        layer_value=attn_value,
                        parent_name="cross-attention layer",
                    )
                )
