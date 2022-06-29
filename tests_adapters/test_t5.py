import unittest

from datasets import load_dataset

from tests.models.t5.test_modeling_t5 import *
from transformers import T5AdapterModel, AutoTokenizer
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, LoRATestMixin, CompacterTestMixin, PrefixTuningTestMixin
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class T5AdapterModelTest(AdapterModelTesterMixin, T5ModelTest):
    all_model_classes = (
        T5AdapterModel,
    )


@require_torch
class T5AdapterTestBase(AdapterTestBase):
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

    def add_head(self, model, name, **kwargs):
        model.add_seq2seq_lm_head(name)
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
            model_inputs = tokenizer(inputs, padding=True, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, padding=True, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        data_args = {
            "task_name": "xsum",
            "path": "./tests/fixtures/tests_samples/xsum/sample.json",
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
class T5AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    T5AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class T5ClassConversionTest(
    ModelClassConversionTestMixin,
    T5AdapterTestBase,
    unittest.TestCase,
):
    pass
