import functools
import inspect
import threading
from typing import ContextManager

from .composition import parse_composition, parse_heads_from_composition


class AdapterSetup(ContextManager):
    """
    Represents an adapter setup of a model including active adapters and active heads. This class is intended to be
    used as a context manager using the ``with`` statement. The setup defined by the ``AdapterSetup`` context will
    override static adapter setups defined in a model (i.e. setups specified via ``active_adapters``).

    Example::

        with AdapterSetup(Stack("a", "b")):
            # will use the adapter stack "a" and "b" outputs = model(**inputs)

    Note that the context manager is thread-local, i.e. it can be used with different setups in a multi-threaded
    environment.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    def __init__(self, adapter_setup, head_setup=None, ignore_empty: bool = False):
        self.adapter_setup = parse_composition(adapter_setup)
        if head_setup:
            self.head_setup = head_setup
        else:
            self.head_setup = parse_heads_from_composition(self.adapter_setup)
        self._empty = ignore_empty and self.adapter_setup is None and self.head_setup is None

    def __enter__(self):
        if not self._empty:
            AdapterSetup.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        if not self._empty:
            AdapterSetup.get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            return None

    @classmethod
    def get_context_adapter_setup(cls):
        context = cls.get_context()
        if context:
            return context.adapter_setup
        return None

    @classmethod
    def get_context_head_setup(cls):
        context = cls.get_context()
        if context:
            return context.head_setup
        return None


class ForwardContext(ContextManager):
    """
    Holds context information during a forward pass through a model. This class should be used via the
    ``ForwardContext.wrap()`` method.

    Note that the context is thread-local.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    context_args = {
        "output_adapter_gating_scores",
        "output_adapter_fusion_attentions",
        "adapter_input_parallelized",
        "task_ids",
    }
    context_attributes = {
        "adapter_gating_scores",
        "adapter_fusion_attentions",
    }
    # Additional used attributes not exposed to the user
    # - prompt_tokens_length: length of the prompt tokens

    def __init__(self, model, *args, **kwargs):
        # If the model has a method ``forward_context()``, use it to create the context.
        for arg_name in self.context_args:
            setattr(self, arg_name, kwargs.pop(arg_name, None))
        if hasattr(model, "forward_context"):
            model.forward_context(self, *args, **kwargs)

    def __enter__(self):
        ForwardContext.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        ForwardContext.get_contexts().pop()

    def _call_forward(self, model, f, *args, **kwargs):
        """
        Calls the forward function of the model with the given arguments and keyword arguments.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in self.context_args}
        results = f(model, *args, **kwargs)

        # append output attributes
        if isinstance(results, tuple):
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results = results + (dict(getattr(self, attr)),)
        else:
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results[attr] = dict(getattr(self, attr))

        return results

    @classmethod
    def add_context_args_in_signature(cls, f):
        old_signature = inspect.signature(f)
        params = list(old_signature.parameters.values())
        # search if a VAR_POSITIONAL or VAR_KEYWORD is present
        # if yes insert step parameter before it, else insert it in last position
        param_types = [param.kind for param in params]
        i = min(
            [
                (param_types.index(param_type) if param_type in param_types else float("inf"))
                for param_type in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            + [len(params)]
        )
        for name in cls.context_args:
            new_param = inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
            if new_param not in params:
                params.insert(i, new_param)
            # we can now build the signature for the wrapper function
        new_signature = old_signature.replace(parameters=params)
        return new_signature

    @classmethod
    def wrap_base(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a base model class.
        Unlike ``wrap()``, this method does not create a new context if the is an existing one.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None and ForwardContext.get_context() is None:
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def wrap(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None:
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            return None
