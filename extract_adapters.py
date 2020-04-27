"""
A simple script that extracts all adapters of a pre-trained model and saves them to separate folders.
"""
import argparse
from transformers.modeling_bert import BertModel
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_xlm_roberta import XLMRobertaModel
from os.path import join
from os import makedirs
from transformers.adapters_utils import get_config_hash


MODELS = {
    "bert": BertModel,
    "roberta": RobertaModel,
    "xlm-roberta": XLMRobertaModel
}


def _get_save_path(model, save_root, model_prefix, adapter_type, name, version, flat):
    # Calculate the hash
    h = get_config_hash(model.adapter_save_config(adapter_type))
    if not flat:
        save_dir = join(save_root, adapter_type + "-" + name, h)
        if version:
            save_dir = join(save_dir, str(version))
        makedirs(save_dir, exist_ok=True)
    else:
        # Build the output folder name
        folder_name = "-".join([model_prefix, str(model.config.hidden_size), adapter_type, name, h[:5]])
        save_dir = join(save_root, folder_name)
        makedirs(save_dir, exist_ok=True)
    return save_dir


def save_all_adapters(model, save_root, prefix, with_head, version=None, flat=False):
    # task adapters
    if hasattr(model.config, 'text_task_adapters'):
        for task_name in model.config.text_task_adapters:
            save_path = _get_save_path(model, save_root, prefix, 'task', task_name, version, flat)
            print("Saving {} adapter to {}...".format(task_name, save_path))
            model.save_task_adapter(save_path, task_name, save_head=with_head)
    # language adapters
    if hasattr(model.config, 'text_lang_adapters'):
        for lang_name in model.config.text_lang_adapters:
            save_path = _get_save_path(model, save_root, prefix, 'lang', lang_name, version, flat)
            print("Saving {} adapter to {}...".format(lang_name, save_path))
            model.save_language_adapter(save_path, lang_name, save_head=with_head)


# python extract_adapters.py --model bert --load-path ../data/Adapters_16_Bert_Base/sst-csqa-hella --save-path ../data/adapters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", help="Model architecture of the loaded model",
                        choices=["bert", "roberta", "xlm-roberta"])
    parser.add_argument("--load-path", type=str, help="Path of the directory containing the model to be loaded.")
    parser.add_argument("--save-path", type=str,
                        help="Path of the directory where the adapters will be saved. Sub-directories are created for all adapters.")
    parser.add_argument("--ver", type=int, default=None, help="Version number of the saved adapters.")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--with-head", action="store_true")

    args = parser.parse_args()

    model_cls = MODELS[args.model]
    print("Loading {} from {}...".format(model_cls.__name__, args.load_path))
    model = model_cls.from_pretrained(args.load_path)

    save_all_adapters(model, args.save_path, args.model, args.with_head, args.ver, args.flat)
