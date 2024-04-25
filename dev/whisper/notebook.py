from datasets import load_dataset, DatasetDict

model_name_or_path = "openai/whisper-large-v2"
language = "Marathi"
language_abbr = "mr"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def convert_features(sample):
    # split inputs and labels since they have to be of different lengths and need different padding methods
    # first treat the audio inputs by simply returning torch tensors

    input_features = [{"input_features": sample["input_features"]}]
    conv_input_features = processor.feature_extractor.pad(input_features, return_tensors="pt")

    # get the tokenized label sequences
    label_features = [{"input_ids": sample["labels"]}]
    # pad the labels to max length
    conv_label_features = processor.tokenizer.pad(label_features, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    conv_label_features = conv_label_features["input_ids"].masked_fill(conv_label_features.attention_mask.ne(1), -100)

    return {"input_features": conv_input_features["input_features"], "labels": conv_label_features}


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])
print(common_voice)
common_voice = common_voice.map(convert_features)
print(common_voice)
