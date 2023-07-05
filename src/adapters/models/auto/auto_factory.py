import importlib

from transformers.models.auto.auto_factory import _LazyAutoMapping, getattribute_from_module, model_type_to_module_name


class _LazyAdapterModelAutoMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "adapters.models")
        return getattribute_from_module(self._modules[module_name], attr)
