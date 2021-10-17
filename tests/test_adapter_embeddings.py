import copy
import tempfile

import torch

from transformers import (
    ADAPTER_CONFIG_MAP,
    MODEL_WITH_HEADS_MAPPING,
    AutoModelWithHeads,
    HoulsbyConfig,
    HoulsbyInvConfig,
    PfeifferConfig,
    PfeifferInvConfig,
)
from transformers.testing_utils import require_torch
import os

@require_torch
class EmbeddingTestMixin:
    def test_load_embedding(self):
        model = self.get_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file = os.path.join(tmp_dir, "embedding.pt")
            torch.save(model.active_embedding[1].weight, file)
            model.load_embedding(file, "test")

        self.assertEqual(model.active_embedding[0], "test")
