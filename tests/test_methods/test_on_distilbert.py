from transformers import DistilBertConfig

from .base import TextAdapterTestBase
from .generator import generate_method_tests
from .method_test_impl.utils import make_config


class DistilBertAdapterTestBase(TextAdapterTestBase):
    config_class = DistilBertConfig
    config = make_config(
        DistilBertConfig,
        dim=32,
        n_layers=4,
        n_heads=4,
        hidden_dim=37,
    )
    tokenizer_name = "distilbert-base-uncased"


method_tests = generate_method_tests(DistilBertAdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
