# coding=utf-8

import json
import logging
import os
import sys
from unittest.mock import patch

import torch

from transformers.testing_utils import TestCasePlus, get_gpu_count, slow, torch_device


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "adapterfusion",
        "text-generation",
        "text-classification",
        "token-classification",
        "language-modeling",
        "multiple-choice",
        "question-answering",
        "summarization",
        "translation",
        "dependency-parsing",
    ]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import run_clm
    import run_fusion_glue
    import run_glue
    import run_mlm
    import run_ner
    import run_qa
    import run_summarization
    import run_swag
    import run_translation
    import run_udp

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


class AdapterExamplesTests(TestCasePlus):
    def test_run_glue_adapters(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_glue.py
            --model_name_or_path bert-base-uncased
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_file ./tests/fixtures/samples/MRPC/train.csv
            --validation_file ./tests/fixtures/samples/MRPC/dev.csv
            --do_train
            --do_eval
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=5e-4
            --max_steps=10
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            --train_adapter
            --adapter_config=double_seq_bn
            """.split()
        with patch.object(sys, "argv", testargs):
            run_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)

    def test_run_fusion_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_fusion_glue.py
            --model_name_or_path bert-base-uncased
            --data_dir ./tests/fixtures/samples/MRPC/
            --task_name mrpc
            --do_train
            --do_eval
            --output_dir ./tests/fixtures/samples/temp_dir
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=5e-5
            --max_steps=20
            --warmup_steps=2
            --overwrite_output_dir
            --seed=42
            --max_seq_length=128
            --train_adapter
            --adapter_config=double_seq_bn
            --load_adapter=qqp@ukp
            """.split()
        with patch.object(sys, "argv", testargs):
            result = run_fusion_glue.main()
            del result["eval_loss"]
            for value in result.values():
                self.assertGreaterEqual(value, 0.5)

    def test_run_squad_adapters(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_squad.py
            --model_name_or_path bert-base-uncased
            --version_2_with_negative
            --train_file ./tests/fixtures/samples/SQUAD/sample.json
            --validation_file ./tests/fixtures/samples/SQUAD/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=15
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-3
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --train_adapter
            --adapter_config=double_seq_bn[reduction_factor=8]
        """.split()

        with patch.object(sys, "argv", testargs):
            run_qa.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_f1"], 30)
            self.assertGreaterEqual(result["eval_exact"], 30)

    def test_run_swag_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_swag.py
            --model_name_or_path bert-base-uncased
            --train_file ./tests/fixtures/samples/swag/sample.json
            --validation_file ./tests/fixtures/samples/swag/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=20
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --train_adapter
            --adapter_config=double_seq_bn[reduction_factor=8]
        """.split()

        with patch.object(sys, "argv", testargs):
            run_swag.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.8)

    def test_run_clm_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_clm.py
            --model_name_or_path gpt2
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --do_train
            --do_eval
            --learning_rate 1e-3
            --block_size 128
            --per_device_train_batch_size 5
            --per_device_eval_batch_size 5
            --num_train_epochs 2
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_adapter
            --adapter_config=double_seq_bn[reduction_factor=8]
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            run_clm.main()
            result = get_results(tmp_dir)
            self.assertLess(result["perplexity"], 100)

    def test_run_mlm_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_mlm.py
            --model_name_or_path roberta-base
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --prediction_loss_only
            --num_train_epochs=1
            --train_adapter
            --adapter_config=double_seq_bn[reduction_factor=8]
        """.split()

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            run_mlm.main()
            result = get_results(tmp_dir)
            self.assertLess(result["perplexity"], 42)

    # TODO: Add Adapter to load
    # def test_generation_adapter(self):
    #     stream_handler = logging.StreamHandler(sys.stdout)
    #     logger.addHandler(stream_handler)
    #
    #     testargs = [
    #         "run_generation.py",
    #         "--prompt=Hello",
    #         "--length=10",
    #         "--seed=42",
    #         "--load_adapter=./test_adapter/adapter_poem",
    #     ]
    #
    #     if is_cuda_and_apex_available():
    #         testargs.append("--fp16")
    #
    #     model_type, model_name = (
    #         "--model_type=gpt2",
    #         "--model_name_or_path=gpt2",
    #     )
    #     with patch.object(sys, "argv", testargs + [model_type, model_name]):
    #         result = run_generation.main()
    #         self.assertGreaterEqual(len(result[0]), 10)

    @slow
    def test_run_summarization_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
                run_summarization.py
                --model_name_or_path facebook/bart-base
                --train_file ./tests/fixtures/samples/xsum/sample.json
                --validation_file ./tests/fixtures/samples/xsum/sample.json
                --output_dir {tmp_dir}
                --overwrite_output_dir
                --max_steps=50
                --warmup_steps=8
                --do_train
                --do_eval
                --learning_rate=2e-4
                --per_device_train_batch_size=2
                --per_device_eval_batch_size=1
                --predict_with_generate
                --train_adapter
                --adapter_config=double_seq_bn[reduction_factor=8]
            """.split()

        with patch.object(sys, "argv", testargs):
            run_summarization.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_rouge1"], 10)
            self.assertGreaterEqual(result["eval_rouge2"], 2)
            self.assertGreaterEqual(result["eval_rougeL"], 7)
            self.assertGreaterEqual(result["eval_rougeLsum"], 7)

    @slow
    def test_run_translation_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
                run_translation.py
                --model_name_or_path facebook/bart-base
                --source_lang en
                --target_lang ro
                --train_file ./tests/fixtures/samples/wmt16/sample.json
                --validation_file ./tests/fixtures/samples/wmt16/sample.json
                --output_dir {tmp_dir}
                --overwrite_output_dir
                --max_steps=50
                --warmup_steps=8
                --do_train
                --do_eval
                --learning_rate=3e-3
                --per_device_train_batch_size=2
                --per_device_eval_batch_size=1
                --predict_with_generate
                --source_lang en_XX
                --target_lang ro_RO
                --train_adapter
                --adapter_config=double_seq_bn[reduction_factor=8]
            """.split()

        with patch.object(sys, "argv", testargs):
            run_translation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_bleu"], 30)

    def test_run_ner_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        # with so little data distributed training needs more epochs to get the score on par with 0/1 gpu
        epochs = 14 if get_gpu_count() > 1 else 6

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_ner.py
            --model_name_or_path bert-base-uncased
            --train_file ./tests/fixtures/samples/conll/sample.json
            --validation_file ./tests/fixtures/samples/conll/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --warmup_steps=2
            --learning_rate=5e-3
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=2
            --num_train_epochs={epochs}
            --train_adapter
            --adapter_config=double_seq_bn[reduction_factor=16]
        """.split()

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            run_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)
            self.assertGreaterEqual(result["eval_precision"], 0.75)
            self.assertLess(result["eval_loss"], 0.5)

    def test_run_udp_adapter(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_udp.py
            --model_name_or_path bert-base-uncased
            --do_train
            --do_eval
            --task_name en_ewt
            --use_mock_data
            --evaluate_on train
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=5e-4
            --max_steps=10
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_adapter
        """.split()

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            run_udp.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_uas"], 100.0)
