import os
import shutil
from pathlib import Path

import datasets
import pandas as pd
import soundfile
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict, load_from_disk

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import Audio


def create_samples():
    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation",
                                use_auth_token=True, streaming=True)
    common_voice = iter(common_voice)

    rows = []
    for i, sample in enumerate(common_voice):
        # path = os.path.join(os.getcwd(), "common_voice", sample["path"])
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # soundfile.write(path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
        rows.append(sample)
        if i == 9:
            break

    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(dataset_path=r'C:\Users\Timo\PycharmProjects\adapters\tests\fixtures\samples\common_voice_en')

    return dataset

create_samples()


common_voice = load_from_disk("common_voice_en")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def convert_to_features(batch):
    input_features = batch["input_features"]
    labels = batch["labels"]
    features = feature_extractor.pad(input_features, return_tensors="pt")
    print(features.shape)


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names, batched=False)
convert_to_features(common_voice)


def train():
    print(len(common_voice[0]["input_features"]))
    print(len(common_voice[0]["input_features"][0]))

    from adapters import WhisperAdapterModel

    model = WhisperAdapterModel.from_pretrained("openai/whisper-small")

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments("test_trainer", num_train_epochs=1, per_device_train_batch_size=2,
                                      logging_dir="logs")
    trainer = Trainer(model=model, args=training_args, train_dataset=common_voice)

    trainer.train()
