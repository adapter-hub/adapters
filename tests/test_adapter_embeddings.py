import os
import tempfile

import torch

from transformers.testing_utils import require_torch


@require_torch
class EmbeddingTestMixin:
    def test_load_embedding(self):
        model = self.get_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_embedding(tmp_dir)
            model.load_embeddings(tmp_dir, "test")

        self.assertEqual(model.active_embedding, "test")

    def test_add_embedding(self):
        model = self.get_model()
        model.add_embeddings("test")
        self.assertEqual(model.active_embedding, "test")

    def test_delete_embedding(self):
        model = self.get_model()
        model.add_embeddings("test")
        self.assertEqual(model.active_embedding, "test")
        model.delete_embeddings("test")
        self.assertFalse("test" in model.loaded_embeddings)
        self.assertEqual(model.active_embedding, "default")

    def test_back_to_default(self):
        model = self.get_model()
        model.eval()
        input_data = self.get_input_samples((1, 128), config=model.config)
        output1 = model(**input_data)

        test_embedding_weights = torch.rand(model.get_input_embeddings().weight.shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(test_embedding_weights, os.path.join(tmp_dir, "embedding.pt"))
            model.load_embeddings(tmp_dir, "test")
        model.set_active_embedding("default")
        output2 = model(**input_data)
        self.assertEqual(model.active_embedding, "default")
        self.assertTrue(torch.equal(output1[0], output2[0]))
