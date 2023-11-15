import argparse
import os

from Utils import (
    compare_lists_close,
    convert_tensors_to_list,
    create_output,
    fix_seeds,
    get_model_names,
    get_new_adapter_config_strings,
    load_model,
    restore_from_jsonl,
)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()


# Create the root path
base_dir = os.path.join(args.path, "model_outputs")
fix_seeds()

for model_name in get_model_names():
    # Load the reference model
    print(f"Model = {model_name}")
    model_dir = os.path.join(base_dir, model_name)
    model = load_model(model_name, os.path.join(model_dir, "model_weights"))

    for adapter_config in get_new_adapter_config_strings():
        # Create a new model output
        adapter_name = model.load_adapter(os.path.join(model_dir, "weights_" + adapter_config))
        model.set_active_adapters(adapter_name)
        model_output = create_output(model, model_name)

        # Compare the model output to the reference output
        model_output_n, last_hidden_state = convert_tensors_to_list(model_output)
        ref_output = restore_from_jsonl(config=adapter_config, file_path=os.path.join(model_dir, "output.jsonl"))
        is_equal = compare_lists_close(ref_output, model_output_n, rtol=1e-05, atol=1e-08)
        print(f"Adapter: {adapter_config} -> {is_equal}")

        model.delete_adapter(adapter_name)
