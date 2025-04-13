from transformers import GPT2Config

from .base import TextAdapterTestBase
from .generator import generate_method_tests
from .method_test_impl.utils import make_config


class GPT2AdapterTestBase(TextAdapterTestBase):
    config_class = GPT2Config
    config = make_config(
        GPT2Config,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=50256,
    )
    tokenizer_name = "gpt2"

    def test_parallel_training_lora(self):
        self.skipTest("Not supported for GPT2")

    def test_mtl_forward_mtl_lora(self):
        self.skipTest("Not supported for GPT2")

    def test_mtl_train_mtl_lora(self):
        self.skipTest("Not supported for GPT2")


method_tests = generate_method_tests(
    GPT2AdapterTestBase,
    not_supported=["PromptTuning", "MTLLoRA", "Vera"],
)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
