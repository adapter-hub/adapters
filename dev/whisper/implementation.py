from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from adapters import init


def forward_pass(model):
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"],
                               return_tensors="pt").input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
init(model)

from adapters import BnConfig

config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
model.add_adapter("bn_adapter", config=config)

from adapters import LoRAConfig

config = LoRAConfig(r=8, alpha=16)
model.add_adapter("lora_adapter", config=config)

from adapters import IA3Config

config = IA3Config()
model.add_adapter("ia3_adapter", config=config)

from adapters import PromptTuningConfig

config = PromptTuningConfig(prompt_length=10)
model.add_adapter("dummy", config=config)

from adapters import ConfigUnion

config = ConfigUnion(
    BnConfig(mh_adapter=True, output_adapter=False, reduction_factor=16, non_linearity="relu"),
    BnConfig(mh_adapter=False, output_adapter=True, reduction_factor=2, non_linearity="relu"),
)
model.add_adapter("union_adapter", config=config)

from adapters import MAMConfig

from adapters import ConfigUnion, ParBnConfig, PrefixTuningConfig

config = ConfigUnion(
    PrefixTuningConfig(bottleneck_size=800),
    ParBnConfig(),
)
#model.add_adapter("mam_adapter", config=config)

model.set_active_adapters("ia3_adapter")


transcription = forward_pass(model)
print(f"Transcription: {transcription}")
