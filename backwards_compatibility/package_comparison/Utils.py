import os
import random
import subprocess
import typing
from subprocess import PIPE
from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from torch import squeeze

import jsonlines
import requests
import transformers
from transformers import (
    AlbertConfig,
    BartConfig,
    BatchEncoding,
    BeitConfig,
    BeitImageProcessor,
    BertConfig,
    CLIPProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    DebertaConfig,
    DebertaV2Config,
    DistilBertConfig,
    EncoderDecoderConfig,
    GPT2Config,
    GPTJConfig,
    MBartConfig,
    RobertaConfig,
    T5Config,
    ViTConfig,
    ViTImageProcessor,
    XLMRobertaConfig,
)


def create_ref_outputs(file_path, venv_python_path, model_name):
    """Create the reference samples for the specified model."""
    # taken from: https://stackoverflow.com/a/27123973
    print(f"venv_python_path = {venv_python_path}")
    print(f"Create reference outputs...")
    process = subprocess.Popen([venv_python_path, file_path, f"--model={model_name}"], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)


def compare_to_ref_output(model_name, rtol=1e-05, atol=1e-05):
    """Compares to reference samples for the specified model."""
    import adapters
    from adapters import AutoAdapterModel
    from transformers import EncoderDecoderModel

    adapter_configs_old, adapter_configs_new = get_adapter_config_strings()
    fix_seeds()

    # Load appropriate directory for comparison
    base_dir_path = os.path.join(os.getcwd(), "Ref_Out")
    model_dir_path = os.path.join(base_dir_path, model_name)
    adapter_dir_path = os.path.join(model_dir_path, "adapters")

    # load saved model
    model_save_dir = os.path.join(model_dir_path, "test_model")
    if model_name == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained(model_save_dir)
        adapters.init(model)
    elif model_name == "encoder_decoder":
        model = EncoderDecoderModel.from_pretrained(model_save_dir)
        adapters.init(model)
    else:
        model = AutoAdapterModel.from_pretrained(model_save_dir)
    model.eval()

    for old, new in zip(adapter_configs_old, adapter_configs_new):
        print(f"config: {new}")
        # create a model instance and add and activate the adatper
        adapter_config_dir = os.path.join(adapter_dir_path, old)
        print(f"adapter_dir = {adapter_config_dir}")
        adapter_name = model.load_adapter(adapter_name_or_path=adapter_config_dir)
        model.set_active_adapters(adapter_name)
        dummy_sample = generate_dummy_data(model=model_name)

        # transfer to device and run forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # dummy_sample.to(device)
        with torch.no_grad():
            model_outputs = model(**dummy_sample)

        # convert the output to a format that can be saved in .jsonl and save it in the model directory
        conv_model_outputs, last_hidden_state = convert_tensors_to_list(model_outputs)

        # compare last hidden state
        file_path_last_hidden_state = os.path.join(model_dir_path, old + ".pt")
        ref = torch.load(file_path_last_hidden_state)
        print(f"Last hidden state equal: {torch.allclose(last_hidden_state, ref, atol=1e-05)}")

        # compare complete model output
        file_path = os.path.join(model_dir_path, model_name + "_outputs.jsonl")
        ref_outputs = restore_from_jsonl(config=old, file_path=file_path)
        compare_lists_close(ref_outputs, conv_model_outputs[0], rtol=rtol, atol=atol)
        model.delete_adapter(adapter_name)



def get_old_adapter_config_strings():
    return [
        "pfeiffer",
        "houlsby",
        "pfeiffer+inv",
        "houlsby+inv",
        "parallel",
        "scaled_parallel",
        "compacter",
        "compacter++",
        "prefix_tuning",
        "prefix_tuning_flat",
        "lora",
        "ia3",
        "mam",
        "unipelt",
    ]
    
def get_new_adapter_config_strings():
    return [
        "seq_bn",
        "double_seq_bn",
        "seq_bn_inv",
        "double_seq_bn_inv",
        "par_bn",
        "scaled_par_bn",
        "compacter",
        "compacter++",
        "prefix_tuning",
        "prefix_tuning_flat",
        "lora",
        "ia3",
        "mam",
        "unipelt",
    ]
    
def get_model_names():
    return [
        "bart",
        "albert",
        "beit",
        "bert",
        "clip",
        "deberta",
        "debertaV2",
        "distilbert",
        "encoder_decoder",
        "gpt2",
        "gptj",
        "mbart",
        "roberta",
        "t5",
        "vit",
        "xlm-r",
    ]

def create_model_instance_without_adapter(model_name: str, model_class: Any) -> Any:
    """Creates and returns an instance of a specified test model.
    Args:
        model_name (str): Specifies which model to instantiate.
    Raises:
        NotImplementedError: If the specified model type is not implemented."""
    from transformers import EncoderDecoderModel

    print(f"model_name = {model_name}")
    if model_name == "bart":
        bart_config = BartConfig(
            d_model=16,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=4,
            decoder_ffn_dim=4,
        )
        print("Create Bart adapter model")
        model = model_class.from_config(bart_config)

    elif model_name == "albert":
        albert_config = AlbertConfig(
            embedding_size=16,
            hidden_size=64,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            num_hidden_groups=2,
        )
        model = model_class.from_config(albert_config)

    elif model_name == "beit":
        beit_config = BeitConfig(
            image_size=224,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
        model = model_class.from_config(beit_config)

    elif model_name == "bert":
        bert_config = BertConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
        model = model_class.from_config(bert_config)

    elif model_name == "clip":
        clip_config = CLIPVisionConfig(
            image_size=30,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
        model = CLIPVisionModelWithProjection(clip_config)

    elif model_name == "deberta":
        deberta_config = DebertaConfig(
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            relative_attention=True,
            pos_att_type="p2c|c2p",
        )
        model = model_class.from_config(deberta_config)

    elif model_name == "debertaV2":
        debertaV2_config = DebertaV2Config(
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            relative_attention=True,
            pos_att_type="p2c|c2p",
        )
        model = model_class.from_config(debertaV2_config)

    elif model_name == "distilbert":
        distilbert_config = DistilBertConfig(
            dim=32,
            n_layers=4,
            n_heads=4,
            hidden_dim=37,
        )
        model = model_class.from_config(distilbert_config)

    elif model_name == "encoder_decoder":
        enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            ),
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
                is_decoder=True,
                add_cross_attention=True,
            ),
        )
        model = EncoderDecoderModel(enc_dec_config)

    elif model_name == "gpt2":
        gpt2_config = GPT2Config(
            n_embd=32,
            n_layer=4,
            n_head=4,
            # set pad token to eos token
            pad_token_id=50256,
        )
        model = model_class.from_config(gpt2_config)

    elif model_name == "gptj":
        gptj_config = GPTJConfig(
            n_embd=32,
            n_layer=4,
            n_head=4,
            rotary_dim=4,
            # set pad token to eos token
            pad_token_id=50256,
            resid_pdrop=0.1,
        )
        model = model_class.from_config(gptj_config)

    elif model_name == "mbart":
        mbart_config = MBartConfig(
            d_model=16,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=4,
            decoder_ffn_dim=4,
            vocab_size=250027,
        )
        model = model_class.from_config(mbart_config)

    elif model_name == "roberta":
        roberta_config = RobertaConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
            vocab_size=50265,
        )
        model = model_class.from_config(roberta_config)

    elif model_name == "t5":
        t5_config = T5Config(
            d_model=16,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            d_ff=4,
            d_kv=16 // 4,
            tie_word_embeddings=False,
            decoder_start_token_id=0,
        )
        model = model_class.from_config(t5_config)

    elif model_name == "vit":
        vit_config = ViTConfig(
            image_size=224,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
        model = model_class.from_config(vit_config)

    elif model_name == "xlm-r":
        xlm_config = XLMRobertaConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
            vocab_size=250002,
        )
        model = model_class.from_config(xlm_config)

    else:
        raise NotImplementedError("The specified model type is not implemented.")

    return model


def add_and_activate_adapter(model: Any, adapter_config: Any, model_name: str) -> Any:
    """Adds and activates an adapter to a given model, returns the name of the adapter.
    Args:
        model (str): Specifies the type of model. Currently supported models are "bart", "bert",
                     "roberta", "gpt2", and "xlm-r".
    Returns:
        Any: The model instance with an active bottleneck adapter."""
    adapter_name = model_name + "_adapter"
    model.add_adapter(adapter_name, config=adapter_config)
    model.set_active_adapters(adapter_name)

    return adapter_name


def generate_dummy_data(model: str = "") -> BatchEncoding:
    """Generates dummy data for text and vision transformer.
    Returns:
        BatchEncoding: A BatchEncoding containing two keys: "input_ids" and
        "attention_mask", each with corresponding tensors containing the dummy data."""
    if model == "beit" or model == "clip" or model == "vit":
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        if model == "beit":
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if model == "clip":
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if model == "vit":
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        return processor(images=image, return_tensors="pt")

    else:
        input_ids = [i for i in range(20)]
        attention_mask = [1 for i in range(len(input_ids))]
        input_ids_tensor = torch.tensor([input_ids])
        attention_mask_tensor = torch.tensor([attention_mask])
        if model == "t5" or model == "encoder_decoder":
            return BatchEncoding(
                {
                    "input_ids": input_ids_tensor,
                    "decoder_input_ids": input_ids_tensor,
                    "attention_mask": attention_mask_tensor,
                }
            )
        return BatchEncoding({"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor})


def fix_seeds(seed: int = 42):
    """Sets seeds manually."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_tuple(tuple_to_decode: tuple):
    """Reconstructs a potentially nested tuple of type `torch.Tensor` as a nested list and returns it.
    Args:
        tuple_to_decode (tuple): The tuple to decode.
    Returns:
        list: A nested list containing the same values as the input tuple.
    Raises:
        TypeError: If the tuple holds values of different type than `tuple` or `torch.Tensor`."""
    inner_model_output = []
    if isinstance(tuple_to_decode, torch.Tensor):
        return tuple_to_decode.cpu().numpy().astype(np.float32).tolist()
    elif isinstance(tuple_to_decode, tuple):
        for value in tuple_to_decode:
            inner_model_output.append(decode_tuple(value))
        return inner_model_output
    else:
        raise TypeError(
            "ERROR occured during decoding of output tensors! The tuple holds values of different type "
            "than `tuple` or `torch.Tensor`."
        )


def convert_tensors_to_list(model_output: transformers.utils.ModelOutput) -> typing.Tuple:
    """Converts the model output, which consists of a Tuple of Tensors to a Tuple of lists, while preserving the
    original dimensions. The converted output is returned.
    Args:
        model_output (transformers.utils.ModelOutput): The model's output of the forward pass.
    Returns:
        list: The converted model output as a list.
    Raises:
        TypeError: If the model output is not a tuple of tensors."""
    # Model ouputs can't be unpacked directly, convert to tuple first
    model_output_tensors = model_output.to_tuple()
    model_output_numpy = []

    # recursively search each tuple entry
    for output_value in model_output_tensors:
        if isinstance(output_value, torch.Tensor):
            model_output_numpy.append(squeeze(output_value.cpu()).numpy().astype(np.float32).tolist())

        elif isinstance(output_value, tuple):
            model_output_numpy.append(decode_tuple(output_value))

    return model_output_numpy, model_output_tensors[0].cpu()


def save_to_jsonl(model_output: list, adapter_config: str, file_path: str):
    """Save model output to .jsonl file as a dictionary. Each line represents one model, where the key is the model
    name and the value the model output stored as a list of lists. If an output for a model is already there it is
    overwritten.
    Args:
        model (str): The model name, serves as the key for the dictionary.
        model_output (list): The model's output as a list.
        file_path (str): The path of the file to save the new entry."""
    # Check if the file exists
    if os.path.exists(file_path):
        # Load content from .jsonl file
        with jsonlines.open(file_path, mode="r") as f:
            data = [line for line in f]
    # Create empty list if file doesn't exist
    else:
        data = []

    # Update result with new one if unique_id already exists in the file
    for i, line in enumerate(data):
        if adapter_config in line:
            data[i] = {adapter_config: model_output}
            break
    # Add new result to the list if unique_id doesn't exist in the file
    else:
        data.append({adapter_config: model_output})
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(data)


def compare_lists_close(a: list, b: list, rtol=1e-05, atol=1e-08):
    """Reimplementation of `allclose()` for lists."""
    # Check if list a and b are numbers
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        bad = abs(a - b) <= atol + rtol * abs(b)
        if not bad:
            print(f"Old package = {a}, New package = {b}")
            print(f"Value diff: {abs(a - b)}")
        return bad

    # Check if a and b are lists
    if isinstance(a, list) and isinstance(b, list):
        # Check if lenghts of the lists are equal
        if len(a) != len(b):
            return False

        for i in range(len(a)):
            if not compare_lists_close(a[i], b[i], rtol=rtol, atol=atol):
                return False

        return True
    # If the inputs are not compatible types
    return False


def restore_from_jsonl(config: str, file_path: str) -> Union[int, list]:
    """-> Restores the model output from a .jsonl file as a list of lists for the specified model.
    Args:
        config (str): Name of the adapter config to restore the output for.
        file_path (str): Path to the .jsonl file containing the model outputs.
    Returns:
        list: A list of lists representing the model output for the specified model. Returns -1 if there is no output for
        the specified model in the file.
    Raises:
        FileExistsError: If the specified file path does not exist."""
    # Check if the file exists
    if os.path.exists(file_path):
        # Load content from .jsonl file
        with jsonlines.open(file_path, mode="r") as f:
            data = [line for line in f]
    else:
        raise FileExistsError(f"There exists no file at the specified path. \npath:{file_path}")
    # Get result of specified model
    for i, line in enumerate(data):
        if config in line:
            return data[i][config]
    else:
        print(f"File does not contain an output for the model {config}.")
        return -1


def save_to_pt(content, file_path):
    torch.save(content, file_path)
