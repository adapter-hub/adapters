from typing import Dict, List, Union

import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk

from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


def create_common_voice():
    """Creates a small abstract dataset of 10 samples from the common voice dataset in english."""
    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation", streaming=True)
    common_voice = iter(common_voice)

    rows = []
    for i, sample in enumerate(common_voice):
        rows.append(sample)
        if i == 9:
            break

    dataset_dict = DatasetDict({"train": Dataset.from_list(rows)})
    dataset_dict.save_to_disk("common_voice_org")
    return dataset_dict


def create_common_voice_encoded(dataset_path="common_voice_org"):
    """Preprocesses the common voice dataset and creates a new encoded version ready for training."""
    model_id = "openai/whisper-tiny"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_rate = feature_extractor.sampling_rate
    decoder_start_token_id = 50257

    # Preprocessing adapted from this example notebook:
    # https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb

    def _prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    def _collate_dataset_with_padding(
        features: List[Dict[str, Union[List[int], torch.Tensor]]], processor, decoder_start_token_id: int
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature} for feature in features["input_features"]]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature} for feature in features["labels"]]
        # pad the labels to max length
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

    dataset = load_from_disk(dataset_path)
    dataset = dataset.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    dataset = dataset.map(_prepare_dataset, remove_columns=dataset.column_names["train"])
    dataset = dataset.map(
        lambda x: _collate_dataset_with_padding(x, processor, decoder_start_token_id),
        batched=True,
        batch_size=10,
    )

    dataset.set_format(type="torch")
    dataset.save_to_disk("common_voice_encoded")


def create_speech_commands():
    """Creates a small abstract dataset of 10 samples from the speech commands dataset."""
    dataset = load_dataset("speech_commands", "v0.02", streaming=True, split="validation")
    labels = [1, 2]

    rows = []
    for i, sample in enumerate(dataset):
        # Assign one of the labels to the sample
        sample["label"] = [labels[i % len(labels)]]
        rows.append(sample)
        if i == 9:
            break

    dataset_dict = DatasetDict({"train": Dataset.from_list(rows)})
    dataset_dict.save_to_disk("speech_commands_org")
    return dataset_dict


def create_speech_commands_encoded(dataset_path="speech_commands_org"):
    """Preprocesses the speech commands dataset and creates a new encoded version ready for training."""
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select_columns(["audio", "label"])

    # Preprocessing copied and adapted from:
    # https://colab.research.google.com/drive/1nU6dlYamT32kfLe2t_AytmOPRjaOxOZn?usp=sharing#scrollTo=GF93pim6eo9e

    model_id = "openai/whisper-tiny"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)

    sampling_rate = feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    max_duration = 30

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
        )
        return inputs

    dataset_encoded = dataset.map(
        preprocess_function,
        remove_columns="audio",
        batched=True,
        batch_size=2,
        num_proc=1,
    )
    # convert to torch format
    dataset_encoded.set_format(type="torch")
    dataset_encoded.save_to_disk("speech_commands_encoded")
    return dataset_encoded


if __name__ == "__main__":

    create_seq2seq = False
    create_classification = False

    if create_seq2seq:
        # Create and preprocess sequence classification dataset
        create_common_voice()
        create_common_voice_encoded()

        # Load and inspect the dataset
        dataset = load_from_disk("common_voice_encoded")
        for sample in dataset["train"]:
            print(sample.keys())
            print(sample["input_features"].shape)
            print(sample["labels"].shape)
            break

    if create_classification:
        # Create and preprocess audio classification dataset
        create_speech_commands()
        create_speech_commands_encoded()

        # Load and inspect the dataset
        dataset = load_from_disk("speech_commands_encoded")
        for sample in dataset["train"]:
            print(sample.keys())
            print(sample["input_features"].shape)
            print(sample["label"])
            break
