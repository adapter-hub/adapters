"""
A simple script that converts older adapter formats to the current format.
"""
import json
from os.path import abspath, join
import torch
import sys
from transformers import AutoModel, CONFIG_MAPPING, CONFIG_NAME, WEIGHTS_NAME


WEIGHTS_CONVERSION_MAP = {
    "layer_adapters": "layer_text_task_adapters",
    "layer_language_adapters": "layer_text_lang_adapters",
    "attention_adapters": "attention_text_task_adapters",
    "language_attention_adapters": "attention_text_lang_adapters"
}

CONFIG_CONVERSION_MAP = {
    "adapter_config": "text_task_adapter_config",
    "language_adapter_config": "text_lang_adapter_config",
    "adapters": "text_task_adapters",
    "language_adapters": "text_lang_adapters"
}


def load_config_from_old_format(model_path):
    with open(join(model_path, CONFIG_NAME), 'r', encoding="utf-8") as f:
        config = json.load(f)
    for k, v in config.items():
        if k in CONFIG_CONVERSION_MAP:
            new_k = CONFIG_CONVERSION_MAP[k]
            config[new_k] = v
            del config[k]
    # load config class
    if "model_type" in config:
        config_class = CONFIG_MAPPING[config["model_type"]]
        return config_class.from_dict(config)
    else:
        for pattern, config_class in CONFIG_MAPPING.items():
            if pattern in model_path:
                return config_class.from_dict(config)
    raise ValueError("Unknown model config class.")


def load_model_from_old_format(model_path):
    old_state_dict = torch.load(join(model_path, WEIGHTS_NAME), map_location="cpu")
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if k in WEIGHTS_CONVERSION_MAP:
            new_k = WEIGHTS_CONVERSION_MAP[k]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    config = load_config_from_old_format(model_path)
    return AutoModel.from_pretrained(None, state_dict=new_state_dict, config=config)


# python convert_model.py ../data/adapters_16_bert_base/sst
if __name__ == "__main__":
    import logging

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    model_path = abspath(sys.argv[1])
    model = load_model_from_old_format(model_path)
    model.save_pretrained(model_path)
