from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, WhisperProcessor
from datasets import load_from_disk, Audio

dataset = load_from_disk("common_voice_en")

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language='English', task="transcribe"
)

# resample audio data from 48 to 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def transform(sample):
    # Load and resample audio data from 48 kHZ to match the model's expected sampling rate
    audio = sample["audio"]

    # compute log-Mel input features from input audio array
    sample["input_features"] = \
        feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors='pt').input_features[0]

    # encode target text to label ids
    sample["labels"] = tokenizer(sample["sentence"]).input_ids
    return sample

def convert_features(sample):
    # split inputs and labels since they have to be of different lengths and need different padding methods
    # first treat the audio inputs by simply returning torch tensors
    new_sample = {}
    input_features = [{"input_features": sample["input_features"]}]
    new_sample["input_features"] = processor.feature_extractor.pad(input_features, return_tensors="pt")["input_features"]

    # get the tokenized label sequences
    label_features = [{"input_ids": sample["labels"]}]
    # pad the labels to max length
    conv_label_features = processor.tokenizer.pad(label_features, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    conv_label_features = conv_label_features["input_ids"].masked_fill(conv_label_features.attention_mask.ne(1), -100)
    new_sample["input_ids"] = conv_label_features
    return new_sample


dataset = dataset.map(transform, batched=False, remove_columns=dataset.column_names)
dataset = dataset.map(convert_features, batched=False, remove_columns=["labels"])
dataset.set_format(type='torch')
print(dataset)
print(dataset[0]["input_features"].shape)

def datacollator(sample):
    pass