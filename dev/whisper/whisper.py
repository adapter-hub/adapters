from adapters import WhisperAdapterModel, ModelAdaptersConfig

model = WhisperAdapterModel.from_pretrained("openai/whisper-tiny")

model.add_seq2seq_lm_head("whisper_head")
model.add_classification_head("whisper_head_classification", num_labels=2)

print(model.heads)
