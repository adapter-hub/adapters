from torch import nn

from .modeling import Adapter, GLOWCouplingBlock, NICECouplingBlock


class InvertibleAdaptersMixin:
    """Mixin for Transformer models adding invertible adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invertible_adapters = nn.ModuleDict(dict())

    def add_invertible_adapter(self, adapter_name: str):
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if adapter_name in self.invertible_adapters:
            raise ValueError(f"Model already contains an adapter module for '{adapter_name}'.")
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[self.config.hidden_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[self.config.hidden_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'.")
            self.invertible_adapters[adapter_name] = inv_adap
            self.invertible_adapters[adapter_name].apply(Adapter.init_bert_weights)

    def delete_invertible_adapter(self, adapter_name: str):
        if adapter_name in self.invertible_adapters:
            del self.invertible_adapters[adapter_name]

    def get_invertible_adapter(self):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                return self.invertible_adapters[first_adapter]
        return None

    def enable_invertible_adapters(self, adapter_names):
        for adapter_name in adapter_names:
            if adapter_name in self.invertible_adapters:
                for param in self.invertible_adapters[adapter_name].parameters():
                    param.requires_grad = True

    def invertible_adapters_forward(self, hidden_states, rev=False):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                hidden_states = self.invertible_adapters[first_adapter](hidden_states, rev=rev)

        return hidden_states
