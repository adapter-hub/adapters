import random

import torch

from transformers import AutoModel, GlueDataset, GlueDataTrainingArguments
from transformers.testing_utils import torch_device


def make_config(config_class, **kwargs):
    return staticmethod(lambda: config_class(**kwargs))


class AdapterTestBase:
    # If not overriden by subclass, AutoModel should be used.
    model_class = AutoModel
    # Default shape of inputs to use
    default_input_samples_shape = (3, 64)

    def get_model(self):
        if self.model_class == AutoModel:
            model = AutoModel.from_config(self.config())
        else:
            model = self.model_class(self.config())
        model.to(torch_device)
        return model

    def get_input_samples(self, shape=None, vocab_size=5000, config=None):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()
        # this is needed e.g. for BART
        if config and config.eos_token_id is not None and config.eos_token_id < vocab_size:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id
        in_data = {"input_ids": input_ids}

        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = input_ids.clone()
        return in_data

    def add_head(self, model, name, **kwargs):
        model.add_classification_head(name, **kwargs)

    def dataset(self, tokenizer):
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        return GlueDataset(data_args, tokenizer=tokenizer, mode="train")
