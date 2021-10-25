import tempfile

import torch

from transformers.testing_utils import require_torch


@require_torch
class EmbeddingTestMixin:
    def test_load_embeddings(self):
        model = self.get_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_embeddings(tmp_dir, "default")
            model.load_embeddings(tmp_dir, "test")

        self.assertEqual(model.active_embeddings, "test")

    def test_add_embeddings(self):
        model = self.get_model()
        model.add_embeddings("test")
        self.assertEqual(model.active_embeddings, "test")

    def test_delete_embeddings(self):
        model = self.get_model()
        model.add_embeddings("test")
        self.assertEqual(model.active_embeddings, "test")
        model.delete_embeddings("test")
        self.assertFalse("test" in model.loaded_embeddings)
        self.assertEqual(model.active_embeddings, "default")

    def test_save_load_embedding(self):
        model = self.get_model()
        input_data = self.get_input_samples((1, 128), config=model.config)

        model.add_embeddings("test")
        model.eval()
        output1 = model(**input_data)
        self.assertEqual(model.active_embeddings, "test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_embeddings(tmp_dir, "test")
            model.load_embeddings(tmp_dir, "test_reloaded")

        self.assertEqual(model.active_embeddings, "test_reloaded")
        output2 = model(**input_data)
        self.assertTrue(
            torch.equal(model.loaded_embeddings["test"].weight, model.loaded_embeddings["test_reloaded"].weight)
        )
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_back_to_default(self):
        model = self.get_model()
        model.eval()
        input_data = self.get_input_samples((1, 128), config=model.config)
        output1 = model(**input_data)

        model.add_embeddings("test")
        self.assertEqual(model.active_embeddings, "test")
        model.set_active_embeddings("default")
        output2 = model(**input_data)
        self.assertEqual(model.active_embeddings, "default")
        self.assertTrue(torch.equal(output1[0], output2[0]))
