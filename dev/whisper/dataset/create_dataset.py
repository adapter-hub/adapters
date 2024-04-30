import os
import shutil
from pathlib import Path

import datasets
import pandas as pd
import soundfile
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict, load_from_disk

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import Audio


def create_samples():
    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation",
                                use_auth_token=True, streaming=True)
    common_voice = iter(common_voice)

    rows = []
    for i, sample in enumerate(common_voice):
        # path = os.path.join(os.getcwd(), "common_voice", sample["path"])
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # soundfile.write(path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
        rows.append(sample)
        if i == 9:
            break

    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(dataset_path=r'/tests/fixtures/samples/common_voice_en')

    return dataset

create_samples()

