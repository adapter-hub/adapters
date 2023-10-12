# https://github.com/google-research/prompt-tuning/blob/main/prompt_tuning/train/prompts.py

import logging
import math
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from .composition import AdapterCompositionBlock, BatchSplit, Parallel, Stack, adjust_tensors_for_parallel
from .configuration import ModelAdaptersConfig, PromptTuningConfig
from .layer import AdapterLayerBase


logger = logging.getLogger(__name__)


Initializer = Callable[[torch.Tensor, Sequence[int]], torch.Tensor]


class PromptTuning(nn.Module):
    """Generate a Prompt and concatenate it with the input.

    This is the training time version of prompting a model. Calling the injected `prompt` module will generate your
    unbatched prompt. This model then replicates it for the batched input and concatenates them together.

    Attributes:
        prompt: The module that actually generates the unbatched prompt.
        combine: A function that combines the prompt and the embedded input.
    """

    prompt: nn.Module
    combination_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        adapter_name: str,
        prompt_tuning_config: PromptTuningConfig,
        model_config: PretrainedConfig,
        base_model_embeddings: nn.Module,
    ):
        super().__init__()

        self.name = adapter_name
        self.model_config = model_config
        self.prompt_tuning_config = prompt_tuning_config
        self.base_model_embeddings = base_model_embeddings

        embedding_size = getattr(model_config, "embedding_size", model_config.hidden_size)

        self.prompt_embedding = nn.Embedding(
            num_embeddings=prompt_tuning_config.prompt_length, embedding_dim=embedding_size
        )
        # Initialize prompt tokens
        self.prompt_tokens = torch.arange(prompt_tuning_config.prompt_length).long()

        self._init_prompt_embedding()

        if prompt_tuning_config.combine == "prefix":
            self.combination_fn = lambda prompt, embedded_input: torch.cat([prompt, embedded_input], dim=1)
        elif prompt_tuning_config.combine == "prefix_after_bos":
            self.combination_fn = lambda prompt, embedded_input: torch.cat(
                [embedded_input[:, 0, np.newaxis], prompt, embedded_input[:, 1:]], dim=1
            )
        else:
            raise ValueError(
                f"Unknown combination function: {prompt_tuning_config.combine}. "
                "Must be one of 'prefix' or 'prefix_after_bos'."
            )

    def _init_prompt_embedding(self) -> None:
        if self.prompt_tuning_config.prompt_init == "random_uniform":
            # Embedding was created using torch.nn.Embedding which already uses a random uniform distribution for initialization
            pass

        elif self.prompt_tuning_config.prompt_init == "from_string":
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer_name_or_path)
            prompt_length = self.prompt_tuning_config.prompt_length
            prompt_text = self.prompt_tuning_config.prompt_init_text
            if prompt_text is None:
                raise ValueError("Prompt text must be provided when using prompt_init='from_string'.")

            tokenized_prompt_text: list[int] = tokenizer(prompt_text)["input_ids"]  # type: ignore

            # If the prompt text tokens are shorter than the prompt length, we repeat the prompt text tokens until we reach the prompt length
            if len(tokenized_prompt_text) < prompt_length:
                num_reps = math.ceil(prompt_length / len(tokenized_prompt_text))
                tokenized_prompt_text = tokenized_prompt_text * num_reps

            # Adjust length of prompt text tokens to match prompt_length
            tokenized_prompt_text = tokenized_prompt_text[:prompt_length]

            # Initialize prompt embedding with tokenized prompt text
            word_embedding_weights = (
                self.base_model_embeddings(torch.LongTensor(tokenized_prompt_text)).detach().clone()
            )
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.prompt_embedding.weight = nn.Parameter(word_embedding_weights)

        else:
            raise ValueError(f"Unknown prompt initialization: {self.prompt_tuning_config.prompt_init}")

    def forward(self, embedded_input):
        # Compute prompt embedding
        self.prompt_tokens = self.prompt_tokens.to(embedded_input.device)
        self.prompt_embedding = self.prompt_embedding.to(embedded_input.device)
        prompt = self.prompt_embedding(self.prompt_tokens)

        # Prompt to batch size
        batch_size = embedded_input.shape[0]
        prompt = torch.tile(torch.unsqueeze(prompt, dim=0), [batch_size] + [1 for _ in prompt.shape])

        # Merge prompt and input
        output = self.combination_fn(prompt, embedded_input)

        # Adapt attention mask
        prefix_attention_mask = torch.ones(batch_size, self.prompt_tuning_config.prompt_length)

        return output, prefix_attention_mask


class PromptTuningLayer(AdapterLayerBase, nn.Module):
    # TODO: add documentation

    def __init__(
        self,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        base_model_embeddings: nn.Module,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.base_model_embeddings = base_model_embeddings
        self.prompt_tunings = nn.ModuleDict()

    def forward(self, hidden_states: torch.Tensor):
        # TODO: Takes currently only very first prompt tuning adapter
        if self.adapters_config.active_setup is not None and len(self.adapters_config.active_setup) > 0:
            first_adapter = self.adapters_config.active_setup.first()
            if first_adapter in self.prompt_tunings:
                hidden_states = self.prompt_tunings[first_adapter](hidden_states)

        return hidden_states

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        # ignore layer_idx as prompt tunings are only added after the embedding layer
        prompt_tuning_config = self.adapters_config.match(
            adapter_name,
            config_type=PromptTuningConfig,
            # layer_idx=self.layer_idx,
            # location_key="prompt_tuning",
        )

        if prompt_tuning_config is not None:
            adapter = PromptTuning(
                adapter_name=adapter_name,
                prompt_tuning_config=prompt_tuning_config,  # type: ignore
                model_config=self.model_config,
                base_model_embeddings=self.base_model_embeddings,
            )
            adapter.train(self.training)  # make sure training mode is consistent
            self.prompt_tunings[adapter_name] = adapter
            return True

        return False

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.prompt_tunings:
            del self.prompt_tunings[adapter_name]

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        pass
        # TODO
        # if unfreeze_adapters:
        #     for prefix_tuning_name in adapter_setup.flatten():
        #         self.pool.enable_prefix(prefix_tuning_name)
        #         if prefix_tuning_name in self.prefix_gates:
        #             for param in self.prefix_gates[prefix_tuning_name].parameters():
        #                 param.requires_grad = unfreeze_adapters

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        pass
        # TODO
        # if adapter_name in self.prefixes:
        #     self.pool.get_prefix(adapter_name)[self.location_key].train(not freeze)
        #     for param in self.pool.get_prefix(adapter_name)[self.location_key].parameters():
        #         param.requires_grad = not freeze
        #     if adapter_name in self.prefix_gates:
        #         for param in self.prefix_gates[adapter_name].parameters():
        #             param.requires_grad = not freeze

    def get_adapter(self, adapter_name):
        # TODO
        # return_dict = nn.ModuleDict()
        # # Make sure to only return params once
        # if adapter_name in self.prefixes and self.prefixes[adapter_name] == 0:
        #     prefix_module = self.pool.get_prefix(adapter_name)
        #     if prefix_module is not None:
        #         return_dict["prefix"] = prefix_module[self.location_key]
        # if adapter_name in self.prefix_gates:
        #     return_dict["gate"] = self.prefix_gates[adapter_name]
        # if len(return_dict) > 0:
        #     return return_dict

        return None

    def average_adapter(self, adapter_name: str, input_adapters: Dict[str, float]) -> bool:
        raise NotImplementedError()

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        raise NotImplementedError()

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        raise NotImplementedError()
