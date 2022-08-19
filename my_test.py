# # Test for Roberta
# import torch
# import os 
# import sys 
# fpath = os.path.join(os.path.dirname(__file__),'src')
# sys.path.append(fpath)
# print(sys.path)


# # from pathlib import Path
# # from transformers import RobertaTokenizer 

# import src.transformers

# # from transformers.configuration_utils import PretrainedConfig
# # from transformers.tokenization_utils import PreTrainedTokenizer
# # from transformers.utils import TensorType
# # from transformers.utils import is_torch_available


# # from src.transformers.models.roberta import RobertaConfig, RobertaTokenizer
# # from src.transformers.adapters import RobertaModelWithHeads

# from src.transformers import (RobertaTokenizer, 
#                           RobertaConfig, 
#                           RobertaModelWithHeads,
#                           TrainingArguments, 
#                           AdapterTrainer, 
#                           EvalPrediction, 
#                           TextClassificationPipeline)
# import src.transformers.adapters.composition as ac
# from src.transformers.adapters.composition import Fuse
# model_ckpt = "roberta-base"
# tokenizer = RobertaTokenizer.from_pretrained(model_ckpt)
# print("hello world")

#========================================================================================

# Test for Roberta
import torch
from datasets import load_dataset
import numpy as np
import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'src')
sys.path.append(fpath)
print(sys.path)


import src.transformers


from src.transformers import (BigBirdTokenizer, 
                          BigBirdConfig, 
                          BigBirdModelWithHeads,
                          RobertaModelWithHeads,
                          TrainingArguments, 
                          AdapterTrainer, 
                          EvalPrediction, 
                          TextClassificationPipeline)


import src.transformers.adapters.composition as ac
from src.transformers.adapters.composition import Fuse
#=================================================================================================

dataset = load_dataset("rotten_tomatoes")

model_ckpt = "google/bigbird-roberta-base"
tokenizer = BigBirdTokenizer.from_pretrained(model_ckpt)


def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

dataset = dataset.map(encode_batch, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# config = BigBirdConfig.from_pretrained(model_ckpt,num_labels=2,)
# model = BigBirdModelWithHeads.from_pretrained(model_ckpt)

model = RobertaModelWithHeads.from_pretrained('roberta-base')
# model = BigBirdModelWithHeads.from_pretrained(model_ckpt)

# # Add a new adapter
# model.add_adapter("rotten_tomatoes")
# # Add a matching classification head
# model.add_classification_head(
#     "rotten_tomatoes",
#     num_labels=2,
#     id2label={ 0: "👎", 1: "👍"}
#   )
# # Activate the adapter
# model.train_adapter("rotten_tomatoes")

#==================================================================================================
# training_args = TrainingArguments(
#     learning_rate=1e-4,
#     num_train_epochs=6,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     logging_steps=200,
#     output_dir="./training_output",
#     overwrite_output_dir=True,
#     # The next line is important to ensure the dataset labels are properly passed to the model
#     remove_unused_columns=False,
# )

# def compute_accuracy(p: EvalPrediction):
#   preds = np.argmax(p.predictions, axis=1)
#   return {"acc": (preds == p.label_ids).mean()}

# trainer = AdapterTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     compute_metrics=compute_accuracy,
# )
#========================================================================================
# trainer.train()
# trainer.evaluate()

# classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# classifier("This is awesome!")