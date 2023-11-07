import os
from Utils import (
    convert_tensors_to_list,
    create_model,
    fix_seeds,
    get_new_adapter_config_strings,
    save_to_jsonl,
    get_model_names,
    create_output,
    load_model,
)

from adapters import AutoAdapterModel, CompacterConfig, CompacterPlusPlusConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()


# Create root path
base_dir = os.path.join(args.path, "model_outputs")
fix_seeds()

for model_name in get_model_names():
    print(f"Model = {model_name}")
    # create dir to contain model- and adapter-weights, and model outputs
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    model = create_model(model_name=model_name, model_class=AutoAdapterModel)
    # save model weights
    model_save_dir = os.path.join(model_dir, "model_weights")
    os.makedirs(model_save_dir, exist_ok=True)
    model.save_pretrained(model_save_dir, from_pt=True)  # save the base model

    for config in get_new_adapter_config_strings():
        # Load model
        model = load_model(model_name, os.path.join(model_dir, "model_weights"))

        # Add adapter
        if config == "compacter++":
            adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        elif config == "compacter":
            adapter_config = CompacterConfig(phm_dim=2, reduction_factor=8)
        else:
            adapter_config = config
        adapter_name = "weights_" + config
        model.add_adapter(adapter_name, config=adapter_config)
        model.set_active_adapters(adapter_name)

        model_output = create_output(model, model_name)

        # process & save output
        model_output_n, last_hidden_state = convert_tensors_to_list(model_output)
        save_to_jsonl(model_output_n, config, os.path.join(model_dir, "output.jsonl"))

        # save adapter weights
        adapter_save_dir = os.path.join(model_dir, adapter_name)
        os.makedirs(adapter_save_dir, exist_ok=True)
        model.save_adapter(save_directory=adapter_save_dir, adapter_name=adapter_name)
        model.delete_adapter(adapter_name)
