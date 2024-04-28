from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from adapters import WhisperAdapterModel, BartAdapterModel

model = WhisperAdapterModel.from_pretrained("openai/whisper-tiny")
print(model)

#bart = BartAdapterModel.from_pretrained("facebook/bart-base")
#print(bart)