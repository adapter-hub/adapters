"""
Code taken and modified from: https://github.com/Adapter-Hub/hgiyt.
Credits: "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models" (Rust et al., 2021)
https://arxiv.org/abs/2012.15613
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

from datasets import load_dataset

import transformers.adapters.composition as ac
from preprocessing import preprocess_dataset
from transformers import (
    AdapterConfig,
    AutoAdapterModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    MultiLingAdapterArguments,
    set_seed,
)
from utils_udp import UD_HEAD_LABELS, DependencyParsingAdapterTrainer, DependencyParsingTrainer, UDTrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    replace_embeddings: bool = field(default=False, metadata={"help": "Whether or not to replace embeddings."})
    leave_out_twelvth: bool = field(
        default=False, metadata={"help": "Whether or not to leave out adapters in twelvth layer"}
    )
    do_lower_case: bool = field(default=False, metadata={"help": "Set this flag when using uncased model/tokenizer"})
    is_japanese: bool = field(default=False, metadata={"help": "Set this to true when using Japanese model/tokenizer"})
    mecab_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to mecab installation. Required when using Japanese model/tokenizer"}
    )
    mecab_dic_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to mecab dictionary. Required when using Japanese model/tokenizer"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The identifier of the Universal Dependencies dataset to train on."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UDTrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare for UD dependency parsing task
    labels = UD_HEAD_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        pad_token_id=-1,
    )

    if model_args.is_japanese:
        assert model_args.mecab_dir is not None
        assert model_args.mecab_dic_dir is not None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
        do_lower_case=model_args.do_lower_case,
        add_prefix_space=True,  # Used e.g. for RoBERTa
        mecab_kwargs={"mecab_option": f"-r {model_args.mecab_dir} -d {model_args.mecab_dic_dir}"}
        if model_args.is_japanese
        else None,
    )

    # The task name (with prefix)
    task_name = "ud_" + data_args.task_name
    language = adapter_args.language

    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.add_dependency_parsing_head(
        task_name,
        num_labels=num_labels,
        id2label=label_map,
    )

    if model_args.leave_out_twelvth:
        logger.info("Leaving out 12")
        leave_out = [11]
    else:
        leave_out = []

    # Setup adapters
    if adapter_args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
                leave_out=leave_out,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                    leave_out=leave_out,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
                leave_out=leave_out,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
                leave_out=leave_out,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        else:
            model.set_active_adapters(task_name)
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    # Load and preprocess dataset
    dataset = load_dataset("universal_dependencies", data_args.task_name)
    dataset = preprocess_dataset(dataset, tokenizer, labels, data_args, pad_token_id=-1)

    # Initialize our Trainer
    # HACK: Set this attribute to False to prevent label columns from being deleted
    training_args.remove_unused_columns = False
    trainer_class = DependencyParsingAdapterTrainer if adapter_args.train_adapter else DependencyParsingTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        logging.info("*** Test ***")

        if training_args.store_best_model:
            logger.info("Loading best model for predictions.")

            if adapter_args.train_adapter:
                if language:
                    lang_adapter_config = AdapterConfig.load(
                        config="pfeiffer", non_linearity="gelu", reduction_factor=2, leave_out=leave_out
                    )
                    model.load_adapter(
                        os.path.join(training_args.output_dir, "best_model", language)
                        if training_args.do_train
                        else adapter_args.load_lang_adapter,
                        config=lang_adapter_config,
                        load_as=language,
                        leave_out=leave_out,
                    )
                task_adapter_config = AdapterConfig.load(
                    config="pfeiffer", non_linearity="gelu", reduction_factor=16, leave_out=leave_out
                )
                model.load_adapter(
                    os.path.join(training_args.output_dir, "best_model", task_name)
                    if training_args.do_train
                    else adapter_args.load_adapter,
                    config=task_adapter_config,
                    load_as=task_name,
                    leave_out=leave_out,
                )
                if language:
                    model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
                else:
                    model.set_active_adapters(task_name)
                model.to(training_args.device)
            else:
                trainer.model = AutoAdapterModel.from_pretrained(
                    os.path.join(training_args.output_dir, "best_model"),
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                ).to(training_args.device)

        predictions, _, metrics = trainer.predict(dataset["test"])

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

    return results


if __name__ == "__main__":
    main()
