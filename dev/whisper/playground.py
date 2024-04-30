from typing import List, Union, Dict
import random
from typing import List, Dict, Union

import datasets
from datasets import Audio
import torch

import adapters
from adapters import AutoAdapterModel
from transformers import AutoFeatureExtractor, AutoTokenizer, GlueDataset, GlueDataTrainingArguments, AutoProcessor
from transformers.testing_utils import torch_device

import torch
from datasets import load_dataset, DatasetDict, load_from_disk, Audio

from adapters import AdapterTrainer, init, SeqBnConfig, WhisperAdapterModel
# transform the dataset

from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor, WhisperModel, TrainingArguments

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = AutoProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def pad_labels(sample):
    # the input_features are already padded by the feature_extractor
    # pad the labels to max length
    padded_labels = processor.tokenizer.pad({"input_ids": [sample["labels"]]}, return_tensors="pt", padding=True)
    print(f"padded_labels: {padded_labels}")

    # replace padding with -100 to ignore loss correctly
    labels = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), -100)

    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        labels = labels[:, 1:]

    sample["labels"] = labels
    return sample




common_voice = DatasetDict()
common_voice["train"] = load_from_disk("dataset/common_voice_en")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])

print(len(common_voice["train"][0]["input_features"]))
print(len(common_voice["train"][0]["labels"]))

common_voice = common_voice.map(pad_labels
print(len(common_voice["train"][0]["labels"][0]))