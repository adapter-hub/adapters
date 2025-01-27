from transformers import T5Config
from transformers.testing_utils import require_torch

from .base import TextAdapterTestBase
from .generator import generate_method_tests
from .method_test_impl.utils import make_config


@require_torch
class T5AdapterTestBase(TextAdapterTestBase):
    config_class = T5Config
    config = make_config(
        T5Config,
        d_model=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=4,
        d_kv=16 // 4,
        tie_word_embeddings=False,
        decoder_start_token_id=0,
    )
    tokenizer_name = "t5-base"


method_tests = generate_method_tests(T5AdapterTestBase, not_supported=["ConfigUnion", "PromptTuning"])
for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
