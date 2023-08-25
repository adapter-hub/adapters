"""
Code taken and modified from: https://github.com/Adapter-Hub/hgiyt.
Credits: "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models" (Rust et al., 2021)
https://arxiv.org/abs/2012.15613
"""
from collections import defaultdict
from typing import List

import datasets
import numpy as np

from transformers import PreTrainedTokenizer


def preprocess_dataset(
    dataset: datasets.DatasetDict,
    tokenizer: PreTrainedTokenizer,
    label_list: List[str],
    data_args,
    pad_token_id=-1,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    def encode_batch(examples):
        features = defaultdict(list)
        for words, heads, deprels in zip(examples["tokens"], examples["head"], examples["deprel"]):
            # clean up
            i = 0
            while i < len(heads):
                if heads[i] == "None":
                    del words[i]
                    del heads[i]
                    del deprels[i]
                i += 1
            tokens = [tokenizer.tokenize(w) for w in words]
            word_lengths = [len(w) for w in tokens]
            tokens_merged = []
            list(map(tokens_merged.extend, tokens))

            if 0 in word_lengths:
                continue
            # Filter out sequences that are too long
            if len(tokens_merged) >= (data_args.max_seq_length - 2):
                continue

            encoding = tokenizer(
                words,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=data_args.max_seq_length,
                is_split_into_words=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )

            input_ids = encoding["input_ids"]
            token_type_ids = encoding["token_type_ids"]
            attention_mask = encoding["attention_mask"]

            pad_item = [pad_token_id]

            # pad or truncate arc labels
            labels_arcs = [int(h) for h in heads]
            labels_arcs = labels_arcs + (data_args.max_seq_length - len(labels_arcs)) * pad_item

            # convert rel labels from map, pad or truncate if necessary
            labels_rels = [label_map[i.split(":")[0]] for i in deprels]
            labels_rels = labels_rels + (data_args.max_seq_length - len(labels_rels)) * pad_item

            # determine start indices of words, pad or truncate if necessary
            word_starts = np.cumsum([1] + word_lengths).tolist()
            word_starts = word_starts + (data_args.max_seq_length + 1 - len(word_starts)) * pad_item

            # sanity check lengths
            assert len(input_ids) == data_args.max_seq_length
            assert len(attention_mask) == data_args.max_seq_length
            assert len(token_type_ids) == data_args.max_seq_length
            assert len(labels_arcs) == data_args.max_seq_length
            assert len(labels_rels) == data_args.max_seq_length
            assert len(word_starts) == data_args.max_seq_length + 1

            features["input_ids"].append(input_ids)
            features["attention_mask"].append(attention_mask)
            features["token_type_ids"].append(token_type_ids)
            features["word_starts"].append(word_starts)
            features["labels_arcs"].append(labels_arcs)
            features["labels_rels"].append(labels_rels)

        return dict(features)

    # Expects columns in all splits to be identical
    remove_columns = dataset.column_names["train"]
    dataset = dataset.map(
        encode_batch,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=remove_columns,
    )
    return dataset
