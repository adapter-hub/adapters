# Dependency parsing on Universal Dependencies with Adapters

These example scripts are based on the fine-tuning code from the repository of ["How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models"](https://github.com/Adapter-Hub/hgiyt).
The scripts were upgraded to `adapters` v2.x and modified to use [flex heads](https://docs.adapterhub.ml/prediction_heads.html#models-with-flexible-heads) and HuggingFace Datasets.

The used biaffine dependency parsing prediction head is described in ["Is Supervised Syntactic Parsing Beneficial for Language Understanding Tasks? An Empirical Investigation" (Glavaš & Vulić, 2021)](https://arxiv.org/pdf/2008.06788.pdf).

A new prediction head can be added to BERT-based models via the `add_dependency_parsing_head()` methods, e.g.:
```python
model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.add_dependency_parsing_head(
    "dependency_parsing",
    num_labels=num_labels,
    id2label=label_map,
)
```

## Training on Universal Dependencies

Script: [`run_udp.py`](https://github.com/Adapter-Hub/adapters/blob/master/examples/dependency-parsing/run_udp.py).

Fine-tuning on the treebanks of [Universal Dependencies](https://universaldependencies.org/).
The datasets are loaded from [HuggingFace Datasets](https://huggingface.co/datasets/universal_dependencies) and which dataset to use can be specified via the `--task_name` option.

Training an adapter on the English Web Treebank (`en_ewt`) could be done as follows:

```bash
export TASK_NAME="en_ewt"

python run_udp.py \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 12 \
    --learning_rate 5e-4 \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score las \
    --train_adapter
```

Fore more information, also visit the original code at https://github.com/Adapter-Hub/hgiyt/tree/master/finetuning.
