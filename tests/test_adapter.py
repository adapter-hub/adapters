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


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()


class AdapterTestBase:
    # If not overriden by subclass, AutoModel should be used.
    model_class = AutoAdapterModel
    # Default shape of inputs to use
    default_input_samples_shape = (3, 64)
    leave_out_layers = [0, 1]
    do_run_train_tests = True
    # default arguments for test_adapter_heads
    batch_size = 1
    seq_length = 128
    is_speech_model = (
        False  # Flag for tests to determine if the model is a speech model due to input format difference
    )

    def get_model(self):
        if self.model_class == AutoAdapterModel:
            model = AutoAdapterModel.from_config(self.config())
        else:
            model = self.model_class(self.config())
            adapters.init(model)
        model.to(torch_device)
        return model

    def get_input_samples(self, shape=None, vocab_size=5000, config=None, **kwargs):
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
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        return GlueDataset(data_args, tokenizer=tokenizer, mode="train")

    def assert_adapter_available(self, model, adapter_name):
        self.assertTrue(adapter_name in model.adapters_config)
        self.assertGreater(len(model.get_adapter(adapter_name)), 0)

    def assert_adapter_unavailable(self, model, adapter_name):
        self.assertFalse(adapter_name in model.adapters_config)
        self.assertEqual(len(model.get_adapter(adapter_name)), 0)


class VisionAdapterTestBase(AdapterTestBase):
    default_input_samples_shape = (3, 3, 224, 224)

    def get_input_samples(self, shape=None, config=None, dtype=torch.float, **kwargs):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        pixel_values = torch.tensor(data=values, dtype=dtype, device=torch_device).view(shape).contiguous()
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
            inputs = feature_extractor([x for x in example_batch["img"]], return_tensors="pt")

            inputs["labels"] = example_batch["label"]
            return inputs

        dataset = datasets.load_dataset(
            "./tests/fixtures/samples/cifar10",
            data_dir="./tests/fixtures/samples/cifar10",
            split="train",
            trust_remote_code=True,
        )
        dataset = dataset.with_transform(transform)

        return dataset


class SpeechAdapterTestBase(AdapterTestBase):
    """Base class for speech adapter tests."""

    default_input_samples_shape = (3, 80, 3000)  # (batch_size, n_mels, enc_seq_len)
    is_speech_model = True  # Flag for tests to determine if the model is a speech model due to input format difference
    time_window = 3000  # Time window for audio samples
    seq_length = 80

    def add_head(self, model, name, head_type="seq2seq_lm", **kwargs):
        """Adds a head to the model."""
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
        """Creates a dummy batch of samples in the format required for speech models."""
        shape = shape or self.default_input_samples_shape

        # Input features
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        input_features = torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
        in_data = {"input_features": input_features}

        # Decoder input ids
        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = ids_tensor((shape[:-1]), config.vocab_size)
        return in_data

    _TASK_DATASET_MAPPING = {
        "seq2seq_lm": "./tests/fixtures/audio_datasets/common_voice_encoded",
        "audio_classification": "./tests/fixtures/audio_datasets/speech_commands_encoded",
    }

    def dataset(self, feature_extractor=None, processor=None, tokenizer=None, task_type: str = "seq2seq_lm", **kwargs):
        """Returns a dataset to test speech model training. Standard dataset is for seq2seq_lm."""
        if task_type == "seq2seq_lm":
            return self._prep_seq2seq_lm_dataset(task_type, **kwargs)
        elif task_type == "audio_classification":
            return self._prep_audio_classification_dataset(task_type, **kwargs)

    def _prep_seq2seq_lm_dataset(self, task_type, **kwargs):
        """Prepares a dataset for conditional generation."""
        # The dataset is already processed and saved to disk, to save time during testing
        # Preparation script can be found in tests/fixtures/audio_datasets/prepare_audio_datasets.py
        dataset_path = self._TASK_DATASET_MAPPING[task_type]
        dataset = datasets.load_from_disk(dataset_path)
        return dataset["train"]

    def _prep_audio_classification_dataset(self, task_type, **kwargs):
        """Prepares a dataset for audio classification."""
        # The dataset is already processed and saved to disk, to save time during testing
        # Preparation script can be found in tests/fixtures/audio_datasets/prepare_audio_datasets.py
        dataset_path = self._TASK_DATASET_MAPPING[task_type]
        dataset = datasets.load_from_disk(dataset_path)
        return dataset["train"]
