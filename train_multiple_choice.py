# REQUIRED: DOWNLOAD SST FROME GLUE $ python transformers/utils/download_glue_data.py --tasks MNLI
from pathlib import Path
import time
import dataclasses
import logging
import os
import sys
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import torch
from torchsummary import summary
from transformers import AutoConfig, BertTokenizer, AutoModelWithHeads, EvalPrediction
from transformers import (Trainer, TrainingArguments, set_seed,)
from transformers import AdapterType, AdapterConfig
from utils_multiple_choice import MultipleChoiceDataset, Split, processors

# Hyperparameters
num_labels = 5
batch_size_per_gpu = 32
max_seq_length = 80
overwrite_tokenizer = False
num_train_epochs = 10
continue_train = False
logging_steps = 100
logging_first_step = True
saving_steps = 1000
learning_rate = 0.00003

# Logging
logging.basicConfig(level=logging.INFO)

# Experiment
base_path = Path("/home/convei_intern1/adapter/")
experiment_name = "hello_mutiple_choice"
task_name="commonsenseqa"

# Data
data_name = "commonsenseqa"
data_path = base_path / Path("data").joinpath(data_name)
tokenizer_path = base_path / Path("tokenizers").joinpath(data_name).joinpath(experiment_name)
logging.info("Data Path: {}".format(data_path))
logging.info("Tokenizer Path: {}".format(tokenizer_path))

# Model
model_path = config_path = base_path / Path("models/"+experiment_name)
if not continue_train:  # model version control
    model_variants = [str(x) for x in Path(base_path / "models").iterdir() if x.is_dir() and str(x).find(experiment_name)]
    model_versions = [ver for var in model_variants for ver in re.findall(r'(\d+)$', var)]
    model_path = config_path = base_path / Path("models/"+experiment_name+str(int(sorted(model_versions)[-1])+1)) if model_versions else model_path
logging.info("Model Path: {}".format(model_path))

# Adapter
adapter_model_path = adapter_config_path = base_path / Path("adapters/"+experiment_name)
if not continue_train:  # adapter version control
    adapter_variants = [str(x) for x in Path(base_path / "adapters").iterdir() if x.is_dir() and str(x).find(experiment_name)]
    adapter_versions = [ver for var in adapter_variants for ver in re.findall(r'(\d+)$', var)]
    adapter_model_path = adapter_config_path = base_path / Path("models/"+experiment_name+str(int(sorted(adapter_versions)[-1])+1)) if adapter_versions else adapter_model_path
logging.info("Adapter Path: {}".format(adapter_model_path))

# Stringfy
data_path = str(data_path)
tokenizer_path = str(tokenizer_path)
config_path = str(config_path)
model_path = str(model_path)
model_name_or_path = str("bert-base-uncased")
adapter_config_name_or_path = str("pfeiffer")
adapter_name_or_path = str("csqa")

# GPU
# REQUIRED: set GPUs to use using shell command
# export CUDA_VISIBLE_DEVICES=0,1,2
logging.info("Current Device Count {}".format(torch.cuda.device_count()))
for did in range(torch.cuda.device_count()):
    logging.info('\t{}'.format(torch.cuda.get_device_name(did)))
device_ids = [0, 1, 2] 
dev = []
for did in device_ids: 
    logging.info('Memory Usage for {} {}'.format(torch.cuda.get_device_name(did), did))
    logging.info('Allocated:{} {}'.format(round(torch.cuda.memory_allocated(did)/1024**3,1), 'GB'))
    logging.info('Cached:   {} {}'.format(round(torch.cuda.memory_reserved(did)/1024**3,1), 'GB'))

# Training
training_args = TrainingArguments(
    logging_first_step=logging_first_step,
    logging_steps=logging_steps,
    per_device_train_batch_size=batch_size_per_gpu,
    per_device_eval_batch_size=batch_size_per_gpu,
    save_steps=-1,
    evaluate_during_training=True,
    output_dir=model_path,
    overwrite_output_dir=continue_train,
    do_train=True,
    do_eval=True,
    do_predict=True,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
)
set_seed(training_args.seed)


# Data Preprocess
tokenizer = BertTokenizer.from_pretrained(
    tokenizer_path,
    cache_dir=tokenizer_path,
)
tokenizer.save_vocabulary(tokenizer_path)
tokenizer.save_pretrained(tokenizer_path)

train_dataset = (
    MultipleChoiceDataset(
        data_dir=data_path,
        tokenizer=tokenizer,
        task=task_name,
        max_seq_length=max_seq_length,
        overwrite_cache=overwrite_tokenizer,
        mode=Split.train,
    )
    if training_args.do_train
    else None
)

eval_dataset = (
    MultipleChoiceDataset(
        data_dir=data_path,
        tokenizer=tokenizer,
        task=task_name,
        max_seq_length=max_seq_length,
        overwrite_cache=overwrite_tokenizer,
        mode=Split.dev,
    )
    if training_args.do_eval
    else None
)


test_dataset = (
    MultipleChoiceDataset(
        data_dir=data_path,
        tokenizer=tokenizer,
        task=task_name,
        max_seq_length=max_seq_length,
        overwrite_cache=overwrite_tokenizer,
        mode=Split.test,
    )
    if training_args.do_eval
    else None
)

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    #cache_dir=cache_dir,
)

model = AutoModelWithHeads.from_pretrained(
    model_name_or_path,
    config=config
)
model.add_multiple_choice_head(adapter_config_name_or_path, num_choices=5)
adapter_config = AdapterConfig.load(adapter_config_name_or_path)
model.load_adapter(adapter_name_or_path, "text_task", config=adapter_config)
model.train_adapter([adapter_name_or_path])
model.set_active_adapters([[adapter_name_or_path]])


# Metric
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}

# summary(model, (batch_size, max_seq_length))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

starttime = time.time()
trainer.train()
trainer.evaluate()
print('Experiment name:', experiment_name)
print('Total time in sec', time.time()-starttime)

Path(model_path).mkdir(parents=True, exist_ok=True)
model.save_pretrained(model_path)

Path(adapter_model_path).mkdir(parents=True, exist_ok=True)
model.save_all_adapters(adapter_model_path)
