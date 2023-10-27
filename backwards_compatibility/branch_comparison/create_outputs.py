
import os
import torch
from Utils import get_new_adapter_config_strings, create_model_instance_without_adapter, get_model_names, generate_dummy_data, fix_seeds
from adapters import AutoAdapterModel

# Create root path
base_dir_path = os.path.join(os.getcwd(), "model_outputs")
print(f"base dir: {base_dir_path}")

# for every model run a forward pass and save the output
fix_seeds()
models_strings = get_model_names()

for model_name in get_model_names():
    print(f"Model: {model_name}")
    # create dir structure for model
    model_dir_path = os.path.join(base_dir_path, model_name)
    model = create_model_instance_without_adapter(model_name=model_name, model_class=AutoAdapterModel)
    model_save_dir = os.path.join(model_dir_path, "test_model")
    model.save_pretrained(model_save_dir, from_pt=True) # save the base model

    for adapter_name in get_new_adapter_config_strings():
        print(f"Adapter: {adapter_name}")
        # for every config create an own dir to save the adapter and output values
        adapter_config_dir = os.path.join(model_dir_path, adapter_name)
        os.makedirs(adapter_config_dir, exist_ok=True)
        
        adapter_name = model_name + "_" + adapter_name
        model.add_adapter(adapter_name, config=adapter_name)
        model.set_active_adapters(adapter_name)
        print(model.active_adapters)
        
        dummy_data = generate_dummy_data(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available
        model.to(device)
        dummy_data.to(device)
        with torch.no_grad():
            model_outputs = model(**dummy_data)
        
        
        # process & save output
        model.save_adapter(save_directory=adapter_config_dir, adapter_name=adapter_name)
        model.delete_adapter(adapter_name)
            