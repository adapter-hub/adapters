import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

from modeling_biaffine import BertForBiaffineParsing
from transformers import (
    AdapterArguments,
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    setup_task_adapter_training,
)
from ud_dataset import Split, UDDataset
from utils_udp import UD_HEAD_LABELS, DependencyParsingTrainer, UDTrainingArguments


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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
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

    data_dir: str = field(metadata={"help": "Path to train, dev, and test data files."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UDTrainingArguments, AdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (model_args, data_args, training_args, adapter_args,) = parser.parse_args_into_dataclasses()

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
        mecab_kwargs={"mecab_option": f"-r {model_args.mecab_dir} -d {model_args.mecab_dic_dir}"}
        if model_args.is_japanese
        else None,
    )

    model = BertForBiaffineParsing.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
    )

    # Setup adapters
    task_name = "udp"
    language = adapter_args.language
    if model_args.replace_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.leave_out_twelvth:
        logger.info("Leaving out 12")
        leave_out = [11]
    else:
        leave_out = []

    setup_task_adapter_training(
        model, task_name, adapter_args, leave_out=leave_out, with_embeddings=model_args.replace_embeddings
    )
    if model_args.leave_out_twelvth:
        if language in model.base_model.encoder.layer._modules["11"].output.layer_text_lang_adapters:
            del model.base_model.encoder.layer._modules["11"].output.layer_text_lang_adapters[language]
            logger.info("Deleted language adapter " + language + " in layer 12")
        if language in model.base_model.encoder.layer._modules["11"].attention.output.attention_text_lang_adapters:
            del model.base_model.encoder.layer._modules["11"].attention.output.attention_text_lang_adapters[language]
            logger.info("Deleted language adapter " + language + " in layer 12")

    if adapter_args.train_adapter:
        if language:
            adapter_names = [[language], [task_name]]
        else:
            adapter_names = [[task_name]]
    else:
        adapter_names = None

    train_dataset = (
        UDDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        UDDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    trainer = DependencyParsingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
        adapter_names=adapter_names,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = UDDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

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
                        AdapterType.text_lang,
                        config=lang_adapter_config,
                        load_as=language,
                    )
                task_adapter_config = AdapterConfig.load(
                    config="pfeiffer", non_linearity="gelu", reduction_factor=16, leave_out=leave_out
                )
                model.load_adapter(
                    os.path.join(training_args.output_dir, "best_model", task_name)
                    if training_args.do_train
                    else adapter_args.load_task_adapter,
                    AdapterType.text_task,
                    config=task_adapter_config,
                    load_as=task_name,
                )
                if model_args.leave_out_twelvth:
                    if language in model.base_model.encoder.layer._modules["11"].output.layer_text_lang_adapters:
                        del model.base_model.encoder.layer._modules["11"].output.layer_text_lang_adapters[language]
                        logger.info("Deleted language adapter " + language + " in layer 12")
                    if (
                        language
                        in model.base_model.encoder.layer._modules["11"].attention.output.attention_text_lang_adapters
                    ):
                        del model.base_model.encoder.layer._modules[
                            "11"
                        ].attention.output.attention_text_lang_adapters[language]
                        logger.info("Deleted language adapter " + language + " in layer 12")

                if language:
                    adapter_names = [[language], [task_name]]
                else:
                    adapter_names = [[task_name]]
            else:
                trainer.model = BertForBiaffineParsing.from_pretrained(
                    os.path.join(training_args.output_dir, "best_model"),
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                ).to(training_args.device)

        predictions, _, metrics = trainer.predict(test_dataset)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

    return results


if __name__ == "__main__":
    main()
