import unittest

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.adapters.composition import Fuse
from transformers.testing_utils import slow


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
            do_save_adapters=True,
            do_save_full_model=False,
        )

        trainer.train()
        # create second model that should resume the training of the first
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.set_active_adapters("adapter")
        trainer_resume = Trainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./examples"),
            train_dataset=train_dataset,
            do_save_adapters=True,
            do_save_full_model=False,
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
            if "adapter" in k1:
                self.assertTrue(torch.equal(v1, v2), k1)

    def test_resume_training_with_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")
        model.add_adapter_fusion(Fuse("adapter", "additional_adapter"))
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
            do_save_adapters=True,
            do_save_full_model=False,
            do_save_adapter_fusion=True,
        )

        trainer.train()
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.add_adapter_fusion(Fuse("adapter", "additional_adapter"))
        model_resume.set_active_adapters(Fuse("adapter", "additional_adapter"))
        trainer_resume = Trainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./examples"),
            train_dataset=train_dataset,
            do_save_full_model=False,
            do_save_adapters=True,
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
            if "adapter" in k1:
                self.assertTrue(torch.equal(v1, v2), k1)

    def test_auto_set_save_adapters(self):
        model = BertForSequenceClassification(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            )
        )
        model.add_adapter("adapter1")
        model.add_adapter("adapter2")
        model.add_adapter_fusion(Fuse("adapter1", "adapter2"))
        model.train_adapter_fusion(Fuse("adapter1", "adapter2"))

        training_args = TrainingArguments(
            output_dir="./examples",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
        )

        self.assertFalse(trainer.do_save_full_model)
        self.assertTrue(trainer.do_save_adapters)
        self.assertTrue(trainer.do_save_adapter_fusion)

    @slow
    def test_training_load_best_model_at_end_full_model(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.train_adapter("adapter")

        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=0.001,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            num_train_epochs=2,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            do_save_adapters=False,
            do_save_full_model=True,
        )

        trainer.train()
        self.assertIsNotNone(trainer.model.active_adapters)


if __name__ == "__main__":
    unittest.main()
