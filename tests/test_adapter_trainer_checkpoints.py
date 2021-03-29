import unittest
from tempfile import TemporaryDirectory

from datasets import load_dataset

from transformers import AutoModelForSequenceClassification, TrainingArguments, GlueDataset, GlueDataTrainingArguments, AutoTokenizer, Trainer


class TestAdapterTrainer(unittest.TestCase):
    def test_resume_training(self):

        TASK = "mrpc"
        dataset = load_dataset("glue", TASK)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(batch["sentence1"], batch["sentence2"], max_length=80, truncation=True,
                             padding="max_length")
        # Encode the input data
        print(dataset["train"][0])
        dataset = dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        dataset.rename_column_("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.set_active_adapters("adapter")

        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=0.1,
            num_train_epochs=3,
            logging_steps=10,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],

        )

        trainer.train()


if __name__ == '__main__':
    unittest.main()
