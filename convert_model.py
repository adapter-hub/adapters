"""
A simple script that converts older adapter formats to the current format.
"""
import json
from os.path import abspath, join
import torch
import sys
from transformers import AutoModel, CONFIG_MAPPING, CONFIG_NAME, WEIGHTS_NAME, AdapterType


WEIGHTS_CONVERSION_MAP = {
    "layer_adapters": "layer_text_task_adapters",
    "layer_language_adapters": "layer_text_lang_adapters",
    "language_attention_adapters": "attention_text_lang_adapters",
    "attention_adapters": "attention_text_task_adapters"
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
    # original format -> newer format w/o ModelAdaptersConfig
    for k, v in config.items():
        if k in CONFIG_CONVERSION_MAP:
            # dirty check to distinguish newest format
            if "adapters" in v:
                continue
            new_k = CONFIG_CONVERSION_MAP[k]
            config[new_k] = v
            del config[k]
    # load config class
    if "model_type" in config:
        config_class = CONFIG_MAPPING[config["model_type"]]
        conf_object = config_class.from_dict(config)
    else:
        for pattern, config_class in CONFIG_MAPPING.items():
            if pattern in model_path:
                conf_object = config_class.from_dict(config)
    # format w/o ModelAdaptersConfig -> newest format
    if conf_object:
        if hasattr(conf_object, "text_lang_adapter_config"):
            conf_object.adapters.set_config(
                AdapterType.text_lang, conf_object.text_lang_adapter_config
            )
            for language in conf_object.text_lang_adapters:
                conf_object.adapters.add(language, AdapterType.text_lang)
            del conf_object.text_lang_adapter_config
            del conf_object.text_lang_adapters
        if hasattr(conf_object, "text_task_adapter_config"):
            conf_object.adapters.set_config(
                AdapterType.text_task, conf_object.text_task_adapter_config
            )
            for task in conf_object.text_task_adapters:
                conf_object.adapters.add(task, AdapterType.text_task)
            del conf_object.text_task_adapter_config
            del conf_object.text_task_adapters
        return conf_object
    else:
        raise ValueError("Unknown model config class.")


def load_model_from_old_format(model_path):
    old_state_dict = torch.load(join(model_path, WEIGHTS_NAME), map_location="cpu")
    new_state_dict = {}
    for k, v in old_state_dict.items():
        # replace all old key names with new ones
        new_k = k
        for o, n in sorted(WEIGHTS_CONVERSION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            new_k = new_k.replace(o, n)
        if k != new_k:
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
