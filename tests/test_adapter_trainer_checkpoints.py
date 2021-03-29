import unittest
from tempfile import TemporaryDirectory

from transformers import AutoModel, TrainingArguments, GlueDataset, GlueDataTrainingArguments, AutoTokenizer, Trainer


class TestAdapterTrainer(unittest.TestCase):
    def test_resume_training(self):
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.add_adapter("test")

        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        training_args = TrainingArguments(
            learning_rate=1e-4,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            logging_steps=100,
            output_dir="./training_output",
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        with TemporaryDirectory() as temp_dir:
            model.save_adapter(temp_dir)
            trainer.train(resume_from_checkpoint=temp_dir)



if __name__ == '__main__':
    unittest.main()
