from transformers.testing_utils import require_torch


@require_torch
class AdapterModelTesterMixin:
    def test_training(self):
        self.skipTest("Not applicable.")

    def test_training_gradient_checkpointing(self):
        self.skipTest("Not applicable.")

    def test_correct_missing_keys(self):
        self.skipTest("Not applicable.")
