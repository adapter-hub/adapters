"""
A simple script that extracts all adapters of a pre-trained model and saves them to separate folders.
"""
import argparse
import json
from transformers.modeling_bert import BertModel
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_xlm_roberta import XLMRobertaModel
from os.path import join, isfile
from os import makedirs
from transformers.adapters_utils import CONFIG_INDEX_FILE, get_config_hash


MODELS = {
    "bert": BertModel,
    "roberta": RobertaModel,
    "xlm-roberta": XLMRobertaModel
}

def _get_save_path(model, save_root, model_prefix, adapter_type, name, suffix, flat):
    # Calculate the hash
    h = get_config_hash(model.adapter_save_config(adapter_type))
    # Build the output folder name
    if not suffix:
        suffix = h[:5]
    folder_name = "-".join([model_prefix, str(model.config.hidden_size), adapter_type, name, suffix])
    if not flat:
        save_dir_base = join(save_root, adapter_type+"-"+name)
        save_dir = join(save_dir_base, folder_name)
        makedirs(save_dir, exist_ok=True)
        # Add the hash to the index file
        index_file = join(save_dir_base, CONFIG_INDEX_FILE)
        if isfile(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_dict = json.load(f)
            index_dict[h] = folder_name
        else:
            index_dict = {h: folder_name}
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_dict, f, indent=2, sort_keys=True)
    else:
        save_dir = join(save_root, folder_name)
        makedirs(save_dir, exist_ok=True)
    return save_dir


def save_all_adapters(model, save_root, prefix, suffix=None, flat=False):
    # task adapters
    if hasattr(model.config, 'adapters'):
        for task_name in model.config.adapters:
            save_path = _get_save_path(model, save_root, prefix, 'task', task_name, suffix, flat)
            print("Saving {} adapter to {}...".format(task_name, save_path))
            model.save_adapter(save_path, task_name)
    # language adapters
    if hasattr(model.config, 'language_adapters'):
        for lang_name in model.config.language_adapters:
            save_path = _get_save_path(model, save_root, prefix, 'lang', lang_name, suffix, flat)
            print("Saving {} adapter to {}...".format(lang_name, save_path))
            model.save_language_adapter(save_path, lang_name)


# python extract_adapters.py --model bert --load-path ../data/Adapters_16_Bert_Base/sst --save-path ../data/adapters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", help="Model architecture of the loaded model",
                        choices=["bert", "roberta", "xlm-roberta"])
    parser.add_argument("--load-path", type=str, help="Path of the directory containing the model to be loaded.")
    parser.add_argument("--save-path", type=str,
        help="Path of the directory where the adapters will be saved. Sub-directories are created for all adapters.")
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--flat", action="store_true")

    args = parser.parse_args()

    model_cls = MODELS[args.model]
    print("Loading {} from {}...".format(model_cls.__name__, args.load_path))
    model = model_cls.from_pretrained(args.load_path)

    save_all_adapters(model, args.save_path, args.model, args.suffix, args.flat)
