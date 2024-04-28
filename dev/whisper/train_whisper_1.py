from dataclasses import dataclass
from typing import Any, List, Union, Dict

import torch
from datasets import DatasetDict, load_from_disk, Audio
from transformers import WhisperFeatureExtractor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperTokenizer, \
    WhisperProcessor, WhisperModel, WhisperForConditionalGeneration

dataset_path = r'C:\Users\timoi\PycharmProjects\adapters\tests\fixtures\samples\common_voice_en'
model_name_or_path = "openai/whisper-small"
language = "English"
task = "transcribe"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prep_featues_and_labels(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def transform(features):
    # split inputs and labels since they have to be of different lengths and need different padding methods
    # first treat the audio inputs by simply returning torch tensors
    input_features = [{"input_features": feature["input_features"]} for feature in features]
    batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

    # get the tokenized label sequences
    label_features = [{"input_ids": feature["labels"]} for feature in features]
    # pad the labels to max length
    labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        labels = labels[:, 1:]

    batch["labels"] = labels

    return batch


def prepare_dataset():
    common_voice = DatasetDict()
    common_voice["train"] = load_from_disk(dataset_path)

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(prep_featues_and_labels, remove_columns=common_voice.column_names["train"])

    return common_voice


common_voice = prepare_dataset()
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

training_args = Seq2SeqTrainingArguments(
    output_dir="./examples",
    do_train=True,
    learning_rate=1.0,
    max_steps=8,
    per_device_train_batch_size=2,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

trainer.train()
