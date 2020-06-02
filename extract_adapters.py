"""
A simple script that extracts all adapters of a pre-trained model and saves them to separate folders.
"""
import argparse
import hashlib
import json
from transformers import AutoModel
from os.path import join, basename
from os import makedirs, listdir
import zipfile
from transformers import get_adapter_config_hash
from convert_model import load_model_from_old_format


def _get_save_path(model, save_root, adapter_type, name, id):
    # Build the output folder name
    folder_name = "-".join(
        [adapter_type, name, model.config.model_type, str(model.config.hidden_size), id]
    )
    save_dir = join(save_root, folder_name)
    makedirs(save_dir, exist_ok=True)
    return save_dir


def save_all_adapters(model, save_root, with_head=False):
    for name in model.config.adapters.adapters:
        adapter_config, adapter_type = model.config.adapters.get(name, return_type=True)
        h = get_adapter_config_hash(adapter_config)
        save_path = _get_save_path(model, save_root, adapter_type, name, h)
        print("Saving {} adapter to {}...".format(name, save_path))
        model.save_adapter(save_path, name, save_head=with_head, meta_dict={'config_id': h})
        yield save_path


def pack_saved_adapters(folders, save_root, version="1"):
    for folder in folders:
        # zip the folder
        folder_name = basename(folder)
        zip_name = join(save_root, "{}.zip".format(folder_name))
        print("Zipping to {}...".format(zip_name))
        zipf = zipfile.ZipFile(zip_name, 'w')
        for file in listdir(folder):
            zipf.write(join(folder, file), arcname=file)
        zipf.close()
        # add description file
        with open(join(folder, 'adapter_config.json'), 'r') as f:
            config = json.load(f)
        # calculate the hash of the zip file
        with open(zip_name, 'rb') as f:
            h = hashlib.sha1(f.read()).hexdigest()
        file_info = {"version": version, "url": "TODO", "sha1": h}
        config['files'] = [file_info]
        config['default_version'] = version
        with open(join(save_root, "{}.json".format(folder_name)), 'w') as f:
            json.dump(config, f, indent=2, sort_keys=True)


# python extract_adapters.py --load-path ../data/Adapters_16_Bert_Base/csqa-multinli-sst --save-path ../data/adapters --from-old
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, required=True, help="Path of the directory containing the model to be loaded.")
    parser.add_argument("--save-path", type=str, required=True,
                        help="Path of the directory where the adapters will be saved. Sub-directories are created for all adapters.")
    parser.add_argument("--ver", type=int, default="1", help="Version number of the saved adapters.")
    parser.add_argument("--with-head", action="store_true")
    parser.add_argument("--from-old", action="store_true", help="Convert from old adapter format.")
    parser.add_argument("--pack", action="store_true", help="Zip the saved files and create description files.")

    args = parser.parse_args()

    print("Loading model from {}...".format(args.load_path))
    if args.from_old:
        model = load_model_from_old_format(args.load_path)
    else:
        model = AutoModel.from_pretrained(args.load_path)

    save_paths = list(save_all_adapters(model, args.save_path, args.with_head))

    if args.pack:
        pack_saved_adapters(save_paths, args.save_path, args.ver)
