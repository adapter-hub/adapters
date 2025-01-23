from transformers import BeitConfig

from .base import VisionAdapterTestBase
from .generator import generate_method_tests
from .method_test_impl.utils import make_config


class BeitAdapterTestBase(VisionAdapterTestBase):
    config_class = BeitConfig
    config = make_config(
        BeitConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = "microsoft/beit-base-patch16-224-pt22k"


method_tests = generate_method_tests(BeitAdapterTestBase, not_supported=["Composition", "Embeddings"])

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
