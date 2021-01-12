# coding=utf-8

import logging
import os
import sys
from unittest.mock import patch

from transformers.testing_utils import TestCasePlus, require_torch_non_multigpu_but_fix_me


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-generation",
        "text-classification",
        "token-classification",
        "language-modeling",
        "question-answering",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_fusion_glue
    import run_glue_alt
    import run_squad


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class AdapterExamplesTests(TestCasePlus):
    @require_torch_non_multigpu_but_fix_me
    def test_run_glue_adapters(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_glue_alt.py
            --model_name_or_path bert-base-uncased
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv
            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv
            --do_train
            --do_eval
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --max_steps=10
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            --train_adapter
            --adapter_config=houlsby
            --load_adapter=qqp@ukp
            """.split()
        with patch.object(sys, "argv", testargs):
            result = run_glue_alt.main()
            del result["eval_loss"]
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)

    @require_torch_non_multigpu_but_fix_me
    def test_run_fusion_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_fusion_glue.py
            --model_name_or_path bert-base-uncased
            --data_dir ./tests/fixtures/tests_samples/MRPC/
            --task_name mrpc
            --do_train
            --do_eval
            --output_dir ./tests/fixtures/tests_samples/temp_dir
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=5e-5
            --max_steps=20
            --warmup_steps=2
            --overwrite_output_dir
            --seed=42
            --max_seq_length=128
            """.split()
        with patch.object(sys, "argv", testargs):
            result = run_fusion_glue.main()
            del result["eval_loss"]
            for value in result.values():
                self.assertGreaterEqual(value, 0.5)

    @require_torch_non_multigpu_but_fix_me
    def test_run_squad_adapters(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_squad.py
            --model_type=bert
            --model_name_or_path=bert-base-uncased
            --data_dir=./tests/fixtures/tests_samples/SQUAD
            --model_name=bert-base-uncased
            --output_dir=./tests/fixtures/tests_samples/temp_dir
            --max_steps=20
            --warmup_steps=2
            --do_train
            --do_eval
            --version_2_with_negative
            --learning_rate=2e-4
            --per_gpu_train_batch_size=2
            --per_gpu_eval_batch_size=1
            --overwrite_output_dir
            --seed=42
            --train_adapter
            --adapter_config=houlsby
            --adapter_reduction_factor=8
        """.split()
        with patch.object(sys, "argv", testargs):
            result = run_squad.main()
            self.assertGreaterEqual(result["f1"], 30)
            self.assertGreaterEqual(result["exact"], 30)
