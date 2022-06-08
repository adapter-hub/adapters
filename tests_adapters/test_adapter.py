import random

import torch
from datasets.commands.dummy_data import MockDownloadManager
import datasets


from transformers import AutoModel, GlueDataset, GlueDataTrainingArguments, AutoTokenizer, AutoFeatureExtractor
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
        return model.heads[name].config["num_labels"]

    def dataset(self, tokenizer=None):
        # setup tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        return GlueDataset(data_args, tokenizer=tokenizer, mode="train")


class VisionAdapterTestBase(AdapterTestBase):
    default_input_samples_shape = (3, 3, 224, 224)

    def get_input_samples(self, shape=None, config=None):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        pixel_values = torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
        in_data = {"pixel_values": pixel_values}

        return in_data

    def add_head(self, model, name, **kwargs):
        if "num_labels" not in kwargs:
            kwargs["num_labels"] = 10
        model.add_image_classification_head(name, **kwargs)
        return model.heads[name].config["num_labels"]

    def dataset(self, feature_extractor=None):
        if feature_extractor is None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_name)

        def transform(example_batch):
            inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

            inputs['labels'] = example_batch['label']
            return inputs

        dataset_builder = datasets.load_dataset_builder("cifar10")

        mock_dl_manager = MockDownloadManager("cifar10", dataset_builder.config, datasets.Version("1.0.0"))
        dataset_builder.download_and_prepare(dl_manager=mock_dl_manager, ignore_verifications=True)

        dataset = dataset_builder.as_dataset(split="train")
        dataset = dataset.with_transform(transform)

        return dataset
