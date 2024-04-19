from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "de", split="train+validation",
                                     use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "de", split="test", use_auth_token=True)

print(common_voice)

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="German", task="transcribe")

print(common_voice["train"][0])