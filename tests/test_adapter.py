import random

import datasets
import torch

import adapters
from adapters import AutoAdapterModel
from transformers import AutoFeatureExtractor, AutoTokenizer, GlueDataset, GlueDataTrainingArguments
from transformers.testing_utils import torch_device


global_rng = random.Random()


def make_config(config_class, **kwargs):
    return staticmethod(lambda: config_class(**kwargs))


class AbstractAdapterTestBase:
    """Base class for adapter tests. Defines basic functions and attributes with default values which are used in the tests.
    Model test classes should inherit from this class or subclass and override the attributes and functions as needed.
    """

    model_class = AutoAdapterModel
    tokenizer_name = "tests/fixtures/SiBERT"  # path to default tokenizer config available in the test repo
    config = None  # specified in the actual model test classes
    input_shape = ()  # (batch_size, seq_length)
    input_shape_generate = ()  # (batch_size, seq_length)
    leave_out_layers = []
    do_run_train_tests = True

    def get_input_samples(self, shape=None, vocab_size=5000, config=None, **kwargs):
        """Creates a dummy batch of samples in the format required for the model."""
        raise NotImplementedError("get_input_samples() must be implemented in the subclass.")

    def add_head(self, model, name, **kwargs):
        """Adds a dummy head to the model."""
        raise NotImplementedError("add_head() must be implemented in the subclass.")

    def get_dataset(self, **kwargs):
        """Loads a dummy dataset for the model."""
        raise NotImplementedError("get_dataset() must be implemented in the subclass.")

    def get_model(self):
        """Builds a model instance for testing based on the provied model configuration."""
        if self.model_class == AutoAdapterModel:
            model = AutoAdapterModel.from_config(self.config())
        else:
            model = self.model_class(self.config())
            adapters.init(model)
        model.to(torch_device)
        return model

    def build_random_tensor(self, shape, dtype=torch.float, **kwargs):
        """Creates a random tensor of the given shape."""
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        if dtype == torch.long and "vocab_size" in kwargs:
            values = [random.randint(0, kwargs["vocab_size"] - 1) for _ in range(total_dims)]
        elif dtype == torch.float:
            values = [random.random() for _ in range(total_dims)]
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        return torch.tensor(data=values, dtype=dtype, device=torch_device).view(shape).contiguous()

    def assert_adapter_available(self, model, adapter_name):
        """Check wether the adapter name is present in the model's adapter config and has been created."""
        self.assertTrue(adapter_name in model.adapters_config)
        self.assertGreater(len(model.get_adapter(adapter_name)), 0)

    def assert_adapter_unavailable(self, model, adapter_name):
        """Check wether the adapter name is not present in the model's adapter config and has not been created."""
        self.assertFalse(adapter_name in model.adapters_config)
        self.assertEqual(len(model.get_adapter(adapter_name)), 0)

    def extract_input_ids(self, inputs):
        # TODO: Check if this is needed in all tests and if it differs between text, vision and speech models
        return inputs["input_ids"]


class TextAdapterTestBase(AbstractAdapterTestBase):
    """Base class for adapter tests for text models. Text models test classes should inherit from this class and override the attributes and functions as needed."""

    input_shape = (3, 64)
    input_shape_generate = (1, 4)
    leave_out_layers = [0, 1]
    batch_size, seq_length = (
        1,
        128,
    )  # TODO: Check in which tests this is needed and if we can simplify by using input_shape

    def get_input_samples(self, shape=None, vocab_size=5000, config=None, **kwargs):
        shape = shape or self.input_shape
        input_ids = self.build_random_tensor(shape, dtype=torch.long)

        # Ensures that only tha last token in each sample is the eos token (needed e.g. for BART)
        if config and config.eos_token_id is not None and config.eos_token_id < vocab_size:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id
        in_data = {"input_ids": input_ids}

        # Add decoder input ids for models with a decoder
        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = input_ids.clone()
        return in_data

    def add_head(self, model, name, **kwargs):
        # TODO: Check if this should be more modular
        model.add_classification_head(name, **kwargs)
        return model.heads[name].config["num_labels"]

    def get_dataset(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        return GlueDataset(data_args, tokenizer=tokenizer, mode="train")


class VisionAdapterTestBase(AbstractAdapterTestBase):
    """Base class for adapter tests for vision models. Vision models test classes should inherit from this class and override the attributes and functions as needed."""

    input_shape = (3, 3, 224, 224)

    def get_input_samples(self, shape=None, config=None, dtype=torch.float, **kwargs):
        shape = shape or self.input_shape
        pixel_values = self.build_random_tensor(shape, dtype=dtype)
        return {"pixel_values": pixel_values}

    def add_head(self, model, name, **kwargs):
        kwargs["num_labels"] = 10 if "num_labels" not in kwargs else kwargs["num_labels"]
        model.add_image_classification_head(name, **kwargs)
        return model.heads[name].config["num_labels"]

    def get_dataset(self, feature_extractor=None):
        dataset = datasets.load_dataset(
            "./tests/fixtures/samples/cifar10",
            data_dir="./tests/fixtures/samples/cifar10",
            split="train",
            trust_remote_code=True,
        )
        if feature_extractor is None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_name)

        def transform(example_batch):
            inputs = feature_extractor([x for x in example_batch["img"]], return_tensors="pt")
            inputs["labels"] = example_batch["label"]
            return inputs

        dataset = dataset.with_transform(transform)
        return dataset


class AudioAdapterTestBase(AbstractAdapterTestBase):
    """Base class for adapter tests for audio models. Audio models test classes should inherit from this class and override the attributes and functions as needed."""

    input_shape = (3, 80, 3000)  # (batch_size, n_mels, enc_seq_len)
    generate_input_shape = (1, 80, 3000)
    time_window = 3000  # Time window for audio samples
    seq_length = 80

    _TASK_DATASET_MAPPING = {
        # TODO: build global mapping for all tasks and datasets
        "seq2seq_lm": "./tests/fixtures/audio_datasets/common_voice_encoded",
        "audio_classification": "./tests/fixtures/audio_datasets/speech_commands_encoded",
    }

    def add_head(self, model, name, head_type="seq2seq_lm", **kwargs):
        # TODO: simpify Audio tests by using the same head type for all tests
        if head_type == "audio_classification":
            model.add_audio_classification_head(name, **kwargs)
            return model.heads[name].config["num_labels"]
        elif head_type == "seq2seq_lm":
            kwargs.pop("num_labels", 1)  # Remove num_labels from kwargs if present in the tests
            model.add_seq2seq_lm_head(name, **kwargs)
            return self.default_input_samples_shape[1]  # Return the number of mel features
        else:
            raise ValueError(f"Head type {head_type} not supported.")

    def get_input_samples(self, shape=None, config=None, **kwargs):
        shape = shape or self.default_input_samples_shape
        in_data = {"input_features": self.build_random_tensor(shape, dtype=torch.float)}

        # Add decoder input ids for models with a decoder
        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = self.build_random_tensor(
                (shape[:-1]), dtype=torch.long, vocab_size=config.vocab_size
            )
        return in_data

    def get_dataset(self, task_type: str = "seq2seq_lm", **kwargs):
        # Dataset is already processed and saved to disk, to save time during testing
        # Preparation script can be found in tests/fixtures/audio_datasets/respective_prepare_script.py
        dataset_path = self._TASK_DATASET_MAPPING[task_type]
        dataset = datasets.load_from_disk(dataset_path)
        return dataset["train"]

    def extract_input_ids(self, inputs):
        return inputs["input_features"]
