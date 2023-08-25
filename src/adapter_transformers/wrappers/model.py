import importlib

from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.models.auto.configuration_auto import model_type_to_module_name

from ..mixins import MODEL_MIXIN_MAPPING
from ..model_mixin import EmbeddingAdaptersWrapperMixin, ModelAdaptersMixin, ModelWithHeadsAdaptersMixin


def wrap_model(model: PreTrainedModel) -> PreTrainedModel:
    if isinstance(model, ModelAdaptersMixin):
        return model

    # First, replace original module classes with their adapter-transformers counterparts
    model_name = model_type_to_module_name(model.config.model_type)
    modules_with_adapters = importlib.import_module(
        f".{model_name}.modeling_{model_name}", "adapter_transformers.models"
    )
    for module in model.modules():
        # Check if module is a base model class
        if module.__class__.__name__ in MODEL_MIXIN_MAPPING:
            # Create new wrapper model class
            model_class = type(
                module.__class__.__name__, (MODEL_MIXIN_MAPPING[module.__class__.__name__], module.__class__), {}
            )
            module.__class__ = model_class
        elif module.__class__.__module__.startswith("transformers.models"):
            try:
                module_class = getattribute_from_module(modules_with_adapters, module.__class__.__name__)
                module.__class__ = module_class
            except ValueError:
                # Silently fail and keep original module class
                pass

    # Next, check if model class itself is not replaced and has an adapter-supporting base class
    if not isinstance(model, ModelAdaptersMixin):
        if hasattr(model, "base_model_prefix") and hasattr(model, model.base_model_prefix):
            base_model = getattr(model, model.base_model_prefix)
            if isinstance(base_model, ModelAdaptersMixin):
                # Create new wrapper model class
                model_class_name = model.__class__.__name__
                model_class = type(
                    model_class_name, (EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin, model.__class__), {}
                )
                model.__class__ = model_class

    # Finally, initialize adapters
    model.init_adapters(model.config)

    return model
