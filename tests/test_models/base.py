from transformers.testing_utils import require_torch


@require_torch
class AdapterModelTesterMixin:
    @property
    def all_generative_model_classes(self):
        return tuple()  # AdapterModel classes are not generative as is (ie without a LM head)

    def test_training(self):
        self.skipTest("Not applicable.")

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        self.skipTest("Not applicable.")

    def test_training_gradient_checkpointing(self):
        self.skipTest("Not applicable.")

    def test_correct_missing_keys(self):
        self.skipTest("Not applicable.")

    def test_generation_tester_mixin_inheritance(self):
        self.skipTest("Not applicable.")  # AdapterModel classes are not generative as is (ie without a LM head)
