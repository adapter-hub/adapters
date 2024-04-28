from typing import List, Dict, Union

import torch
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, WhisperProcessor, WhisperModel
from datasets import load_from_disk, Audio

dataset = load_from_disk("common_voice_en")

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language='English', task="transcribe")



def transform(sample):
    # Load and resample audio data from 48 kHZ to match the model's expected sampling rate
    audio = sample["audio"]

    # compute log-Mel input features from input audio array
    sample["input_features"] = \
        feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors='pt').input_features[0]

    # encode target text to label ids
    sample["labels"] = tokenizer(sample["sentence"]).input_ids
    return sample


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
    print(sentences)
    labels = tokenizer(sentences).input_ids

    # return the batch
    batch["input_features"] = input_features
    batch["labels"] = labels
    return batch


def convert_features(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    # split inputs and labels since they have to be of different lengths and need different padding methods
    # first treat the audio inputs by simply returning torch tensors
    batched_i_f = features["input_features"]
    input_features = [{"input_features": feature} for feature in batched_i_f]
    batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

    # get the tokenized label sequences
    labels_batched = features["labels"]
    lables = [{"input_ids": label} for label in labels_batched]
    # pad the labels to max length
    padded_labels = processor.tokenizer.pad(lables, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    labels = padded_labels["input_ids"].masked_fill(padded_labels.attention_mask.ne(1), -100)

    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        labels = labels[:, 1:]

    batch["labels"] = labels
    batch["decoder_input_ids"] = labels
    return batch


def non_batched(dataset):
    dataset = dataset.map(transform)
    dataset = dataset.map(convert_features, batched=True)
    dataset.set_format(type="torch", columns=["input_features", "labels"])
    print(dataset[0])


def batched(dataset):
    # Resampling audio from 48 kHZ to match the model's expected sampling rate, executed upon reading the dataset
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.map(convert_features, batched=True)


    dataset.set_format(type="torch", columns=["input_features", "labels", "decoder_input_ids"])
    return dataset


dataset = batched(dataset)
print(type(dataset))

def training(dataset):
    # Training

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="./examples",
        do_train=True,
        learning_rate=1.0,
        max_steps=8,
        per_device_train_batch_size=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=WhisperModel.from_pretrained("openai/whisper-small"),
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
