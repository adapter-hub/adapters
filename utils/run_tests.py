"""
Runs adapter tests and a subset of other tests relevant for adapter-transformers.
"""
import pytest


TESTED_MODULES = [
    "test_adapter",
    "test_modeling_auto",
    "test_modeling_bart",
    "test_modeling_bert",
    "test_modeling_distilbert",
    "test_modeling_gpt2",
    "test_modeling_mbart",
    "test_modeling_roberta",
    "test_modeling_xlm_roberta",
    "test_modeling_encoder_decoder",
    "test_modeling_t5",
    "test_trainer",
]


if __name__ == "__main__":
    test_selection = " or ".join(TESTED_MODULES)
    args = [
        "-k",
        test_selection,
        "--numprocesses=auto",
        "--dist=loadfile",
        "-s",
        "-v",
        "--ignore-glob=tests/test_tokenization*",
        "--ignore-glob=tests/test_processor*",
        "./tests",
    ]
    exit_code = pytest.main(args)
    exit(exit_code)
