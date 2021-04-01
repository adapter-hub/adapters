import copy
import unittest
from tempfile import TemporaryDirectory

import torch
from datasets import load_dataset

from transformers import AutoModel, TrainingArguments, GlueDataset, GlueDataTrainingArguments, AutoTokenizer, Trainer, \
    AutoModelForSequenceClassification
from transformers.adapter_composition import Fuse


class TestAdapterTrainer(unittest.TestCase):
    def test_resume_training(self):

        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(batch["sentence1"], batch["sentence2"], max_length=80, truncation=True,
                             padding="max_length")

        TASK = "mrpc"
        dataset = load_dataset("glue", TASK)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Encode the input data
        print(dataset["train"][0])
        dataset = dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        dataset.rename_column_("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")
        model.set_active_adapters("adapter")

        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=0.1,
            logging_steps=1,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
        )

        trainer.train()
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.set_active_adapters("adapter")
        trainer_resume = Trainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./examples"),
            train_dataset=dataset["train"]
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.equal(v1, v2), k1)

    def test_resume_training_with_fusion(self):
        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(batch["sentence1"], batch["sentence2"], max_length=80, truncation=True,
                             padding="max_length")

        TASK = "mrpc"
        dataset = load_dataset("glue", TASK)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Encode the input data
        print(dataset["train"][0])
        dataset = dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        dataset.rename_column_("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")
        model.add_fusion(Fuse("adapter", "additional_adapter"))
        model.set_active_adapters(Fuse("adapter", "additional_adapter"))

        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=0.1,
            logging_steps=1,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
        )

        trainer.train()
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.add_fusion(Fuse("adapter", "additional_adapter"))
        model_resume.set_active_adapters(Fuse("adapter", "additional_adapter"))
        trainer_resume = Trainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./examples"),
            train_dataset=dataset["train"]
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.equal(v1, v2), k1)


if __name__ == '__main__':
    unittest.main()
