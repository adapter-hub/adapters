import os
import tempfile

import torch

from transformers import T5ForConditionalGeneration
from transformers.testing_utils import require_torch


@require_torch
class EmbeddingTestMixin:
    def test_load_embedding(self):
        model = self.get_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_embedding(tmp_dir)
            model.load_embedding(tmp_dir, "test")

        self.assertEqual(model.active_embedding, "test")

    def test_back_to_default(self):
        model = self.get_model()
        model.eval()
        input_data = self.get_input_samples((1, 128), config=model.config)
        output1 = model(**input_data)

        test_embedding_weights = torch.rand(model.get_embedding_module().weight.shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(test_embedding_weights, os.path.join(tmp_dir, "embedding.pt"))
            model.load_embedding(tmp_dir, "test")
        model.set_active_embedding("default")
        output2 = model(**input_data)
        self.assertEqual(model.active_embedding, "default")
        self.assertTrue(torch.equal(output1[0], output2[0]))
