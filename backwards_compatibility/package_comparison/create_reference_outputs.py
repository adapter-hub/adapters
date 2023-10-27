import argparse
import os

import torch

from transformers import CLIPVisionModelWithProjection, EncoderDecoderModel
from transformers.adapters import AutoAdapterModel, CompacterConfig, CompacterPlusPlusConfig
from backwards_compatibility.package_comparison.Utils import (
    add_and_activate_adapter,
    convert_tensors_to_list,
    create_model_instance_without_adapter,
    fix_seeds,
    generate_dummy_data,
    get_adapter_config_strings,
    save_to_jsonl,
    save_to_pt,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()
model_name = args.model

adapter_configs, _ = get_adapter_config_strings()
fix_seeds()

# Create directory structure for saving model output and adapter weight
base_dir_path = os.path.join(os.getcwd(), "Ref_Out")
model_dir_path = os.path.join(base_dir_path, model_name)
adapter_dir_path = os.path.join(model_dir_path, "adapters")
output_file_path = os.path.join(model_dir_path, model_name + "_outputs.jsonl")

model = create_model_instance_without_adapter(model_name=model_name, model_class=AutoAdapterModel)
# save the model
model_save_dir = os.path.join(model_dir_path, "test_model")
model.save_pretrained(model_save_dir, from_pt=True)

for config in adapter_configs:
    print(f"config: {config}")
    # for every config create an own dir for adapter weights and output values
    adapter_config_dir = os.path.join(adapter_dir_path, config)
    os.makedirs(adapter_config_dir, exist_ok=True)

    # Get model and create dummy data
    if model_name == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained(model_save_dir)
    elif model_name == "encoder_decoder":
        model = EncoderDecoderModel.from_pretrained(model_save_dir)
    else:
        model = AutoAdapterModel.from_pretrained(model_save_dir)
    model.eval()
    if config == "compacter++":
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        adapter_name = add_and_activate_adapter(model=model, adapter_config=adapter_config, model_name=model_name)
    elif config == "compacter":
        adapter_config = CompacterConfig(phm_dim=2, reduction_factor=8)
        adapter_name = add_and_activate_adapter(model=model, adapter_config=adapter_config, model_name=model_name)
    else:
        adapter_name = add_and_activate_adapter(model=model, adapter_config=config, model_name=model_name)
    dummy_sample = generate_dummy_data(model=model_name)

    # transfer to device and run forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_sample.to(device)
    with torch.no_grad():
        model_outputs = model(**dummy_sample)

    # convert the output to a format that can be saved in .jsonl and save it in the model directory
    conv_model_outputs, last_hidden_state = convert_tensors_to_list(model_outputs)
    file_path_last_hidden_state = os.path.join(model_dir_path, config + ".pt")
    save_to_pt(
        content=last_hidden_state, file_path=file_path_last_hidden_state
    )  # save last hidden state seperately for double checking
    save_to_jsonl(adapter_config=config, model_output=conv_model_outputs[0], file_path=output_file_path)

    # save the adapter in the respective config subdir
    model.save_adapter(save_directory=adapter_config_dir, adapter_name=adapter_name)
    model.delete_adapter(adapter_name)
