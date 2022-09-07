# # Test for Roberta
# import torch
# from datasets import load_dataset
# import numpy as np
# import os 
# import sys 
# fpath = os.path.join(os.path.dirname(__file__),'src')
# sys.path.append(fpath)
# print(sys.path)


# import src.transformers


# from src.transformers import (BigBirdTokenizer, 
#                           BigBirdConfig, 
#                           BigBirdModelWithHeads,
#                           RobertaModelWithHeads,
#                           TrainingArguments, 
#                           Trainer, 
#                           AdapterTrainer,
#                           EvalPrediction, 
#                           TextClassificationPipeline,
#                           RobertaTokenizer,
#                           BigBirdForSequenceClassification
# )



# import src.transformers.adapters.composition as ac
# from src.transformers.adapters.composition import Fuse
# # from src.transformers.adapters import AdapterTrainer
# #=================================================================================================

# dataset = load_dataset("rotten_tomatoes")

# model_ckpt = "google/bigbird-roberta-base"

# # model_ckpt = "roberta-base"
# tokenizer = BigBirdTokenizer.from_pretrained(model_ckpt)

# # tokenizer = RobertaTokenizer.from_pretrained(model_ckpt)

# def encode_batch(batch):
#   """Encodes a batch of input data using the model tokenizer."""
#   return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# dataset = dataset.map(encode_batch, batched=True)
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# config = BigBirdConfig.from_pretrained(model_ckpt,num_labels=2,)

# # model = RobertaModelWithHeads.from_pretrained(model_ckpt)

# # #=====================================================================================================
# model = BigBirdModelWithHeads.from_pretrained(model_ckpt,config=config)
# # model = BigBirdForSequenceClassification.from_pretrained(model_ckpt,config=config)

# print("after model initialization")

# # Add a new adapter
# model.add_adapter("rotten_tomatoes1")

# # Add a matching classification head
# model.add_classification_head(
#     "rotten_tomatoes1",
#     num_labels=2,
#     id2label={ 0: "👎", 1: "👍"}
#   )
# # Activate the adapter
# model.train_adapter("rotten_tomatoes1")

# # model.train()

# print("Hey there we are ready with the model!!")

# training_args = TrainingArguments(
#     learning_rate=1e-4,
#     num_train_epochs=1,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     logging_steps=200,
#     output_dir="./training_output",
#     overwrite_output_dir=True,
#     # The next line is important to ensure the dataset labels are properly passed to the model
#     remove_unused_columns=False,
# )

# # # AdapterTrainer is throwing error : temporarily tested with Trainer
# # print("Now error will come=========================")

# def compute_accuracy(p: EvalPrediction):
#   print(p.predictions)
#   print(p.label_ids)

#   preds = np.argmax(p.predictions, axis=1)
#   return {"acc": (preds == p.label_ids).mean()}

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     # compute_metrics=compute_accuracy,
# )

# trainer.train()
# trainer.evaluate()
# print("Training Done !!")

# #Inference
# sentence = "This is awesome!"
# tokens = tokenizer(sentence)
# input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
# outputs = model(input_ids)
# print(outputs.logits)

# # model.save_adapter("./adapter1", "rotten_tomatoes1")

# # #=================================================================================

# # model = BigBirdModelWithHeads.from_pretrained(model_ckpt,config=config)

# # # Add a new adapter
# # model.add_adapter("rotten_tomatoes2")

# # # Add a matching classification head
# # model.add_classification_head(
# #     "rotten_tomatoes2",
# #     num_labels=2,
# #     id2label={ 0: "👎", 1: "👍"}
# #   )
# # # Activate the adapter
# # model.train_adapter("rotten_tomatoes2")

# # print("Hey there we are ready with the model!!")

# # training_args = TrainingArguments(
# #     learning_rate=1e-4,
# #     num_train_epochs=1,
# #     per_device_train_batch_size=32,
# #     per_device_eval_batch_size=32,
# #     logging_steps=200,
# #     output_dir="./training_output",
# #     overwrite_output_dir=True,
# #     # The next line is important to ensure the dataset labels are properly passed to the model
# #     remove_unused_columns=False,
# # )

# # # AdapterTrainer is throwing error : temporarily tested with Trainer

# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=dataset["train"],
# #     eval_dataset=dataset["validation"],
# # )

# # trainer.train()
# # trainer.evaluate()
# # print("Training Done !!")

# # #Inference
# # sentence = "This is awesome!"
# # tokens = tokenizer(sentence)
# # input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
# # outputs = model(input_ids)
# # print(outputs.logits)

# # model.save_adapter("./adapter2", "rotten_tomatoes2")
# # #==============================================================================================
# # print("Fusion Starting ...")
# # print("========================================================================================================")

# # model = BigBirdModelWithHeads.from_pretrained(model_ckpt,config=config)

# # model.load_adapter("./adapter1", source="hf", load_as="adapter1", with_head=False)
# # model.load_adapter("./adapter2", source="hf", load_as="adapter2", with_head=False)

# # # Add a fusion layer for all loaded adapters
# # model.add_adapter_fusion(Fuse("adapter1", "adapter2"))
# # model.set_active_adapters(Fuse("adapter1", "adapter2"))

# # # Add a classification head for our target task
# # model.add_classification_head("rotten_tomatoes_head",num_labels=2,id2label={ 0: "👎", 1: "👍"})

# # # Unfreeze and activate fusion setup
# # adapter_setup = Fuse("adapter1", "adapter2")
# # model.train_adapter_fusion(adapter_setup)


# # training_args = TrainingArguments(
# #     learning_rate=1e-4,
# #     num_train_epochs=1,
# #     per_device_train_batch_size=32,
# #     per_device_eval_batch_size=32,
# #     logging_steps=200,
# #     output_dir="./training_output",
# #     overwrite_output_dir=True,
# #     # The next line is important to ensure the dataset labels are properly passed to the model
# #     remove_unused_columns=False,
# # )

# # # AdapterTrainer is throwing error : temporarily tested with Trainer

# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=dataset["train"],
# #     eval_dataset=dataset["validation"],
# # )

# # # Model Training
# # trainer.train()

# # model.save_adapter_fusion("./output_fusion",Fuse("adapter1", "adapter2"))
# # model.save_head("./output_fusion_head","rotten_tomatoes_head")


# # # Add a fusion layer for all loaded adapters
# # model.load_adapter_fusion("./output_fusion")

# # model.load_head("./output_fusion_head")

# # model.set_active_adapters(Fuse("adapter1", "adapter2"))

# # #Inference
# # sentence = "This is awesome!"
# # tokens = tokenizer(sentence)
# # input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
# # outputs = model(input_ids)
# # print(outputs.logits)

# Test for Roberta
#==========================================================================================================================
import torch
from datasets import load_dataset
import numpy as np
import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'src')
sys.path.append(fpath)
print(sys.path)


from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")

from transformers import BigBirdTokenizer

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

from transformers import BigBirdConfig, BigBirdModelWithHeads

config = BigBirdConfig.from_pretrained(
    "google/bigbird-roberta-base",
    num_labels=2,
)
model = BigBirdModelWithHeads.from_pretrained(
    "google/bigbird-roberta-base",
    config=config,
)

# Add a new adapter
model.add_adapter("rotten_tomatoes")
# Add a matching classification head
model.add_classification_head(
    "rotten_tomatoes",
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"}
  )
# Activate the adapter
model.train_adapter("rotten_tomatoes")

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)

trainer.train()

trainer.evaluate()

from transformers import TextClassificationPipeline

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

print(classifier("This is awesome!"))
