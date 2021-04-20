import unittest

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
<<<<<<< HEAD
from transformers.adapter_composition import Fuse
=======
from transformers.adapters.composition import Fuse
>>>>>>> v2


class TestAdapterTrainer(unittest.TestCase):
    def test_resume_training(self):

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

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
            train_dataset=train_dataset,
<<<<<<< HEAD
        )

        trainer.train()
=======
            do_save_adapters=True,
            do_save_full_model=False,
        )

        trainer.train()
        # create second model that should resume the training of the first
>>>>>>> v2
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.set_active_adapters("adapter")
        trainer_resume = Trainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./examples"),
            train_dataset=train_dataset,
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
<<<<<<< HEAD
            self.assertTrue(torch.equal(v1, v2), k1)
=======
            if "adapter" in k1:
                self.assertTrue(torch.equal(v1, v2), k1)
>>>>>>> v2

    def test_resume_training_with_fusion(self):
        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(
                batch["sentence1"], batch["sentence2"], max_length=80, truncation=True, padding="max_length"
            )

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

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
            train_dataset=train_dataset,
<<<<<<< HEAD
=======
            do_save_adapters=True,
            do_save_full_model=False,
            do_save_adapter_fusion=True,
>>>>>>> v2
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
            train_dataset=train_dataset,
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
<<<<<<< HEAD
            self.assertTrue(torch.equal(v1, v2), k1)
=======
            if "adapter" in k1:
                self.assertTrue(torch.equal(v1, v2), k1)
>>>>>>> v2


if __name__ == "__main__":
    unittest.main()
