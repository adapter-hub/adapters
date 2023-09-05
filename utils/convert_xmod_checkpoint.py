"""
This script can be used to convert an Xmod checkpoints (including adapters) from the HF format to the Adapters format.
"""
import argparse
import os
import re

import torch

from adapters import SeqBnConfig, XmodAdapterModel
from transformers import XmodModel


def convert_xmod_checkpoint(model_name: str, output_dir: str):
    # Instantiate new model
    orig_model = XmodModel.from_pretrained(model_name)
    model_config = orig_model.config
    new_model = XmodAdapterModel.from_pretrained(model_name)
    for lang in model_config.languages:
        adapter_config = SeqBnConfig(
            reduction_factor=model_config.adapter_reduction_factor,
            # selection between (shared) adapter LN and original LN is done in XmodOutput
            original_ln_before=model_config.adapter_layer_norm or model_config.adapter_reuse_layer_norm,
            original_ln_after=False,
            residual_before_ln=False if model_config.ln_before_adapter else "post_add",
            non_linearity=model_config.hidden_act,
        )
        new_model.add_adapter(lang, adapter_config)

    # Convert state dict
    new_state_dict = {}
    for k, v in orig_model.state_dict().items():
        if match := re.match(r"(.+)\.adapter_modules\.(?P<lang>\w+)\.(?P<layer>\w+)\.(.+)", k):
            prefix, suffix = match.group(1, 4)
            lang = match.group("lang")
            layer = match.group("layer")
            if layer == "dense1":
                new_layer = "adapter_down.0"
            elif layer == "dense2":
                new_layer = "adapter_up"
            else:
                raise ValueError(f"Unknown layer {layer}")
            new_k = f"{new_model.base_model_prefix}.{prefix}.adapters.{lang}.{new_layer}.{suffix}"
            new_state_dict[new_k] = v
        else:
            new_state_dict[f"{new_model.base_model_prefix}.{k}"] = v
    missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Check equal outputs
    orig_model.eval()
    new_model.eval()
    inputs = orig_model.dummy_inputs
    for lang in model_config.languages:
        orig_model.set_default_language(lang)
        orig_outputs = orig_model(**inputs)
        new_model.set_active_adapters(lang)
        new_outputs = new_model(**inputs)
        all_close = torch.allclose(orig_outputs.last_hidden_state, new_outputs.last_hidden_state)
        check_str = "OK" if all_close else "FAIL"
        print(f"{lang:>6}: {check_str}")

    # Save new model & all adapters
    os.makedirs(output_dir, exist_ok=True)
    new_model.save_all_adapters(output_dir)
    # Remove all adapters except for English
    for lang in model_config.languages:
        if lang != "en_XX":
            new_model.delete_adapter(lang)
    new_model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()

    convert_xmod_checkpoint(args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
