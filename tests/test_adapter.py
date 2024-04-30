import random
from typing import List, Dict, Union

import datasets
from datasets import Audio
import torch

import adapters
from adapters import AutoAdapterModel
from transformers import AutoFeatureExtractor, AutoTokenizer, GlueDataset, GlueDataTrainingArguments, AutoProcessor
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

    def get_model(self):
        if self.model_class == AutoAdapterModel:
            model = AutoAdapterModel.from_config(self.config())
        else:
            model = self.model_class(self.config())
            adapters.init(model)
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
            inputs = feature_extractor([x for x in example_batch["img"]], return_tensors="pt")

            inputs["labels"] = example_batch["label"]
            return inputs

        dataset = datasets.load_dataset(
            "./tests/fixtures/samples/cifar10",
            data_dir="./tests/fixtures/samples/cifar10",
            split="train",
        )
        dataset = dataset.with_transform(transform)

        return dataset


class SpeechAdapterTestBase(AdapterTestBase):
    default_input_samples_shape = (1, 80, 3000)

    def dataset(self, feature_extractor=None, processor=None, tokenizer=None):
        if feature_extractor is None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_name)

        if processor is None:
            processor = AutoProcessor.from_pretrained(self.processor_name)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Preprocessing functions adapted from this example notebook:
        # https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb

        def prepare_dataset(batch):
            # Load and resample audio data from 48 kHZ to match the model's expected sampling rate
            audio = batch["audio"]
            raw_speech = [sample["array"] for sample in audio]
            sampling_rate = audio[0]["sampling_rate"]

            # compute log-Mel input features from input audio array
            input_features = feature_extractor(raw_speech=raw_speech, sampling_rate=sampling_rate,
                                               return_tensors='pt').input_features

            # encode target text to label ids
            sentences = batch["sentence"]
            labels = tokenizer(sentences).input_ids

            # return the batch
            batch["input_features"] = input_features
            batch["input_ids"] = labels
            return batch

        def convert_features(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            batched_i_f = features["input_features"]
            input_features = [{"input_features": feature} for feature in batched_i_f]
            batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            labels_batched = features["input_ids"]
            lables = [{"input_ids": label} for label in labels_batched]
            # pad the labels to max length
            padded_labels = processor.tokenizer.pad(lables, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["input_ids"] = labels

            return batch

        # Load an extract of 10 samples from the Common Voice dataset
        dataset = datasets.load_from_disk(dataset_path="./tests/fixtures/samples/common_voice_en")

        # Resampling audio from 48 kHZ to match the model's expected sampling rate, executed upon reading the dataset
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

        # Preprocessing the dataset
        dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.map(convert_features, batched=True)
        dataset.set_format(type="torch", columns=["input_features", "input_ids"])
        print(dataset)

        return dataset

    def get_input_samples(self, shape=None, config=None):
        """ Creates and returns  a dict with the key 'input_features' containing a random tensor of shape `shape`.
        The method is used for creating a test speech sample for speech models which require the key 'input_features'
        in the input dict instead of 'input_ids'. """
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        input_features = torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
        in_data = {"input_features": input_features}
        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = ids_tensor((shape[0:2]), config.vocab_size)
        return in_data
