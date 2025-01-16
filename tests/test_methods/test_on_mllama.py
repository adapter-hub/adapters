import random

from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaTextConfig, MllamaVisionConfig

from .generator import *


def from_text_vision_configs(config_class, text_config: MllamaTextConfig, vision_config: MllamaVisionConfig, **kwargs):
    """
    Create a MllamaConfig instance from separate text and vision configs.

    This standalone function mimics the behavior of class methods like CLIPConfig.from_text_vision_configs,
    but works without modifying the MllamaConfig class.

    Args:
        config_class: The configuration class to instantiate (MllamaConfig)
        text_config: The configuration for the text model
        vision_config: The configuration for the vision model
        **kwargs: Additional arguments to pass to the config constructor

    Returns:
        An instance of config_class initialized with the text and vision configs
    """
    return config_class(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


class MllamaAdapterTestBase(TextAdapterTestBase):

    config = staticmethod(
        lambda: from_text_vision_configs(
            MllamaConfig,
            MllamaTextConfig(
                vocab_size=1000,  # Minimal vocab size
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                intermediate_size=256,
                cross_attention_layers=[0],
                bos_token_id=990,
                eos_token_id=991,
                pad_token_id=992,
                max_position_embeddings=512,
                rope_scaling={
                    "rope_type": "default",
                },
            ),
            MllamaVisionConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_global_layers=1,
                num_attention_heads=1,
                intermediate_size=256,
                vision_output_dim=128,
                image_size=112,
                patch_size=4,
            ),
        )
    )
    tokenizer_name = "arnavgrg/mllama-11b-vision-lora"

    def get_input_samples(self, vocab_size=5000, config=None, dtype=torch.float, **kwargs):
        # text inputs
        shape = self.default_text_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()
        # this is needed e.g. for BART
        if config and config.eos_token_id is not None and config.eos_token_id < vocab_size:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id
        in_data = {"input_ids": input_ids}

        # vision inputs
        shape = self.default_vision_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        pixel_values = torch.tensor(data=values, dtype=dtype, device=torch_device).view(shape).contiguous()
        in_data["pixel_values"] = pixel_values

        return in_data


test_methods = generate_method_tests(MllamaAdapterTestBase, excluded_tests=[])

for test_class_name, test_class in test_methods.items():
    globals()[test_class_name] = test_class
