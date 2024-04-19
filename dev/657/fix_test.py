from transformers import BertModel, BertTokenizer
from adapters import ConfigUnion, LoRAConfig, PrefixTuningConfig, SeqBnConfig, Fuse, init

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
init(model)

config = ConfigUnion(LoRAConfig(r=8, use_gating=True), PrefixTuningConfig(prefix_length=10, use_gating=True))
model.add_adapter("unipelt", config=config)
seq_config = SeqBnConfig(reduction_factor=16, use_gating=True)
model.add_adapter("adapter1", config=seq_config)  # Adapter1
model.add_adapter("adapter2", config=seq_config)  # Adapter2
model.add_adapter("adapter3", config=seq_config)  # Adapter3
adapter_setup = Fuse("adapter1", "adapter2", "adapter3")
model.add_adapter_fusion(adapter_setup)  # Adapter fusion
model.set_active_adapters([adapter_setup, 'adapter1', 'unipelt'])


def run_model():
    sample = "Hello, my dog is cute"
    inputs = tokenizer(sample, return_tensors="pt")
    outputs = model(**inputs)
    return outputs


print(run_model())
