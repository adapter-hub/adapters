import unittest

from datasets import load_dataset

from transformers import AutoTokenizer, BertGenerationConfig
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    PromptTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class BertGenerationAdapterTestBase(AdapterTestBase):
    config_class = BertGenerationConfig
    config = make_config(
        BertGenerationConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "bert-base-uncased"

    def add_head(self, model, name, **kwargs):
        model.add_masked_lm_head(name)
        return self.default_input_samples_shape[-1]

    def dataset(self, tokenizer=None):
        # setup tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            inputs = examples["document"]
            targets = examples["summary"]
            inputs = ["Summarize: " + inp for inp in inputs]
            model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        data_args = {
            "task_name": "xsum",
            "path": "./tests/fixtures/samples/xsum/sample.json",
        }
        dataset = load_dataset("json", data_files=data_args["path"])
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset",
        )
        return train_dataset


@require_torch
class BertGenerationAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    PromptTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    BertGenerationAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BertGenerationClassConversionTest(
    ModelClassConversionTestMixin,
    BertGenerationAdapterTestBase,
    unittest.TestCase,
):
    pass
