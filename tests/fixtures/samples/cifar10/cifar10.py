"""
CIFAR-10 demo data, adapted from https://huggingface.co/datasets/cifar10.
"""
import os
import pickle

import datasets
import numpy as np


class Cifar10(datasets.GeneratorBasedBuilder):
    """CIFAR-10 Data Set"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of CIFAR-10 Data Set",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "img": datasets.Image(),
                    "label": datasets.features.ClassLabel(num_classes=10),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": ["test_batch"], "split": "test"},
            ),
        ]

    def _generate_examples(self, files, split):
        for file in files:
            with open(os.path.join(self.config.data_dir, file), "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")

                labels = dict[b"labels"]
                images = dict[b"data"]

                for idx, _ in enumerate(images):

                    img_reshaped = np.transpose(np.reshape(images[idx], (3, 32, 32)), (1, 2, 0))

                    yield f"{file}_{idx}", {
                        "img": img_reshaped,
                        "label": labels[idx],
                    }
