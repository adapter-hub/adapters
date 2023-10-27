from adapters import AutoAdapterModel, CompacterConfig

from transformers import BertConfig

bert_config = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )

model = AutoAdapterModel.from_config(bert_config)

config = CompacterConfig()
model.add_adapter("test", config=config)