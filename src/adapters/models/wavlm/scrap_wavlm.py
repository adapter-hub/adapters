from src.adapters.models import wavlm

model = wavlm.WavLMAdapterModel.from_pretrained("microsoft/wavlm-large")
print(model)