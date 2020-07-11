import glob
import hashlib
import json
import zipfile
from argparse import ArgumentParser, Namespace
from os import listdir, makedirs
from os.path import basename, dirname, join
from typing import List

import ruamel.yaml
from PyInquirer import prompt

from transformers import WEIGHTS_NAME, AutoModel, BertForSequenceClassification, get_adapter_config_hash
from transformers.adapter_utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.adapter_utils import download_cached
from transformers.commands import BaseTransformersCLICommand


ADAPTER_TEMPLATE_YAML = "https://raw.githubusercontent.com/calpt/nothing-to-see-here/master/TEMPLATES/adapter.template.yaml"

ADAPTER_KEYS_TO_COPY = ["type", "model_type", "model_name", "model_class"]


def adapter_pack_command_factory(args: Namespace):
    return AdapterPackCommand(args.input_path, args.output_path, args.template)


class AdapterPackCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        extract_parser = parser.add_parser(
            "pack",
            help="CLI tool extract all adapters from models in a directory and preparing them for upload to AdapterHub."
        )
        extract_parser.add_argument("input_path", type=str, help="Path to a directory pretrained models or pretrained adapters.")
        extract_parser.add_argument("-o", "--output_path", type=str, help="Path to a directory where the packed adapters will be saved. By default, the root of the input path is used.")
        extract_parser.add_argument("--template", type=str, help="Path to a YAML file to be used as template for the adapter info cards.")
        extract_parser.set_defaults(func=adapter_pack_command_factory)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        template: str,
    ):
        self.input_path = input_path
        self.output_path = output_path or input_path
        self.template = template

    def find_models_and_adapters(self) -> List:
        # full models and adapters are identified by their weights file
        models_list = []
        for weights_file in glob.glob(join(self.input_path, "**", WEIGHTS_NAME), recursive=True):
            models_list.append(dirname(weights_file))
        adapters_list = []
        for weights_file in glob.glob(join(self.input_path, "**", ADAPTER_WEIGHTS_NAME), recursive=True):
            adapters_list.append(dirname(weights_file))
        return models_list, adapters_list

    def pack_adapters(self, folders, save_root, version="1"):
        if not self.template:
            template_file = download_cached(ADAPTER_TEMPLATE_YAML)
        else:
            template_file = self.template
        for folder in folders:
            # zip the folder
            # TODO
            config_name = "pfeiffer" if "pfeiffer" in folder else "houlsby"
            folder_name = "_".join(["bert-base-uncased", basename(folder), config_name])
            zip_name = join(save_root, "{}.zip".format(folder_name))
            print("Zipping {} to {}...".format(folder, zip_name))
            zipf = zipfile.ZipFile(zip_name, "w")
            for file in listdir(folder):
                zipf.write(join(folder, file), arcname=file)
            zipf.close()
            # add description file
            with open(join(folder, "adapter_config.json"), "r") as f:
                config = json.load(f)
            # TODO
            config['model_name'] = "bert-base-uncased"
            # calculate the hash of the zip file
            with open(zip_name, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()
            file_info = {"version": version, "url": "TODO", "sha1": h}
            # load the template
            yaml = ruamel.yaml.YAML()
            with open(template_file, 'r') as f:
                template = yaml.load(f)
            for key in ADAPTER_KEYS_TO_COPY:
                if key in config:
                    template[key] = config[key]
            template["files"] = [file_info]
            template["default_version"] = version
            template["config"] = {"using": config_name}
            with open(join(save_root, "{}.yaml".format(folder_name)), "w") as f:
                yaml.dump(template, f)

    def run(self):
        models_list, adapters_list = self.find_models_and_adapters()
        print("=" * 10 + " Found models and adapters " + "=" * 10)
        for model_dir in models_list:
            print(f"Model in {model_dir}")
        for adapter_dir in adapters_list:
            print(f"Adapter in {adapter_dir}")
        answers = prompt([{
            "type": "confirm",
            "name": "continue",
            "message": "Extract and pack all listed adapters:",
            "default": False
        }])
        if not answers['continue']:
            return
        # extract adapters from all models
        for model_dir in models_list:
            print(f"Extracting adapters from model in {model_dir} ...")
            # TODO
            model = BertForSequenceClassification.from_pretrained(model_dir)
            for adapter_path in model.save_all_adapters(model_dir):
                if adapter_path not in adapters_list:
                    adapters_list.append(adapter_path)
        self.pack_adapters(adapters_list, self.output_path)
