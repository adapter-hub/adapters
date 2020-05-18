"""
A simple script that extracts all adapters of a pre-trained model and saves them to separate folders.
"""
import argparse
from transformers import BertModel, RobertaModel, XLMRobertaModel, AdapterType
from os.path import join
from os import makedirs
from transformers import get_adapter_config_hash
from convert_model import load_model_from_old_format


MODELS = {
    "bert": BertModel,
    "roberta": RobertaModel,
    "xlm-roberta": XLMRobertaModel
}


def _get_save_path(model, save_root, model_prefix, adapter_type, name, id, version, flat):
    if not flat:
        save_dir = join(save_root, adapter_type, name, id)
        if version:
            save_dir = join(save_dir, str(version))
        makedirs(save_dir, exist_ok=True)
    else:
        # Build the output folder name
        folder_name = "-".join([model_prefix, str(model.config.hidden_size), adapter_type, name, id[:5]])
        save_dir = join(save_root, folder_name)
        makedirs(save_dir, exist_ok=True)
    return save_dir


def save_all_adapters(model, save_root, prefix, with_head, version=None, flat=False):
    for name in model.config.adapters.adapters:
        adapter_type = model.config.adapters.get_type(name)
        h = get_adapter_config_hash(model.adapters[adapter_type].full_config())
        save_path = _get_save_path(model, save_root, prefix, adapter_type, name, h, version, flat)
        print("Saving {} adapter to {}...".format(name, save_path))
        model.save_adapter(save_path, name, save_head=with_head, meta_dict={'id': h})


# python extract_adapters.py --model bert --load-path ../data/Adapters_16_Bert_Base/csqa-multinli-sst --save-path ../data/adapters --from-old
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
    parser.add_argument("--from-old", action="store_true", help="Convert from old adapter format.")

    args = parser.parse_args()

    model_cls = MODELS[args.model]
    print("Loading {} from {}...".format(model_cls.__name__, args.load_path))
    if args.from_old:
        model = load_model_from_old_format(args.load_path)
    else:
        model = model_cls.from_pretrained(args.load_path)

    save_all_adapters(model, args.save_path, args.model, args.with_head, args.ver, args.flat)
