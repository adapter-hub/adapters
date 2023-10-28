import os
import torch
from Utils import convert_tensors_to_list ,save_to_jsonl, get_new_adapter_config_strings, create_model, get_model_names, generate_dummy_data, fix_seeds, create_output, save_model_output
from adapters import AutoAdapterModel, init

# Create root path
base_dir = os.path.join(os.getcwd(), "model_outputs")
fix_seeds()

for model_name in get_model_names():
    
    print(f"Model = {model_name}")
    model_dir = os.path.join(base_dir, model_name)
    model = create_model(model_name=model_name, model_class=AutoAdapterModel)
    init(model)

    for adapter_config in get_new_adapter_config_strings():
                
        # create model output
        adapter_name = "weights"+adapter_config
        model, model_output = create_output(model, model_name, adapter_config, adapter_name)
        
        # process & save output
        model_output_n, last_hidden_state = convert_tensors_to_list(model_output)
        save_to_jsonl(model_output_n, adapter_config, os.path.join(model_dir, "output.jsonl"))
        
        # save adapter weights
        adapter_save_dir = os.path.join(model_dir, adapter_name)
        os.makedirs(adapter_save_dir, exist_ok=True)
        model.save_adapter(save_directory=adapter_save_dir, adapter_name=adapter_name)
        model.delete_adapter(adapter_name)

    # save model weights
    model_save_dir = os.path.join(model_dir, "model_weights")
    os.makedirs(model_save_dir, exist_ok=True)
    model.save_pretrained(model_save_dir, from_pt=True) # save the base model
