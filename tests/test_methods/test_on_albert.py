import unittest
from math import ceil

import pytest

from transformers import AlbertConfig
from transformers.testing_utils import require_torch

from .base import TextAdapterTestBase
from .generator import generate_method_tests
from .method_test_impl.heads.test_adapter_heads import PredictionHeadModelTestMixin
from .method_test_impl.utils import make_config


class AlbertAdapterTestBase(TextAdapterTestBase):
    """Model configuration for testing methods on Albert."""

    config_class = AlbertConfig
    config = make_config(
        AlbertConfig,
        embedding_size=16,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        num_hidden_groups=2,
    )
    tokenizer_name = "albert-base-v2"
    leave_out_layers = [0]


method_tests = generate_method_tests(AlbertAdapterTestBase, not_supported=["Heads"])

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class


@require_torch
@pytest.mark.heads
class Heads(
    AlbertAdapterTestBase,
    PredictionHeadModelTestMixin,
    unittest.TestCase,
):

    def test_context_simple(self):
        expected_number_of_adapter_calls = ceil(self.config().num_hidden_layers / self.config().num_hidden_groups)
        super().test_context_simple(expected_number_of_adapter_calls=expected_number_of_adapter_calls)
