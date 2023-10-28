from adapters import AutoAdapterModel, CompacterConfig, init

from transformers import BertConfig, CLIPVisionConfig, CLIPVisionModelWithProjection

bert_config = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )

model = AutoAdapterModel.from_config(bert_config)

config = CompacterConfig()
model.add_adapter("test", config=config)


clip_config = CLIPVisionConfig(
            image_size=30,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
model = CLIPVisionModelWithProjection(clip_config)

init(model)

model.add_adapter("test", "compacter")