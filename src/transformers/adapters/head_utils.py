import logging
import re


logger = logging.getLogger(__name__)


STATIC_TO_FLEX_HEAD_MAP = {
    # BERT
    "BertForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": ["classifier"],
    },
    "BertForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": ["classifier"],
    },
    "BertForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["classifier"],
    },
    "BertForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["qa_outputs"],
    },
    # RoBERTa
    "RobertaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
            "use_pooler": False,
        },
        "layers": ["classifier.dense", "classifier.out_proj"],
    },
    "RobertaForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": ["classifier"],
    },
    "RobertaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["classifier"],
    },
    "RobertaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["qa_outputs"],
    },
    # XLM-RoBERTa
    "XLMRobertaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
            "use_pooler": False,
        },
        "layers": ["classifier.dense", "classifier.out_proj"],
    },
    "XLMRobertaForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": ["classifier"],
    },
    "XLMRobertaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["classifier"],
    },
    "XLMRobertaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["qa_outputs"],
    },
    # BART
    "BartForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": ["classification_head.dense", "classification_head.out_proj"],
    },
    "BartForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["qa_outputs"],
    },
    # MBART
    "MBartForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": ["classification_head.dense", "classification_head.out_proj"],
    },
    "MBartForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["qa_outputs"],
    },
    # DistilBERT
    "DistilBertForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "relu",
        },
        "layers": ["pre_classifier", "classifier"],
    },
    "DistilBertForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 2,
            "activation_function": "relu",
        },
        "layers": ["pre_classifier", "classifier"],
    },
    "DistilBertForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["classifier"],
    },
    "DistilBertForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["qa_outputs"],
    },
    # GPT-2
    "GPT2ForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "bias": False,
        },
        "layers": ["score"],
    },
}


def _regex_list_rename_func(k, rename_list):
    for o, n in rename_list:
        match = re.match(o, k)
        if match:
            return n.format(match.group(1))
    return k


def get_head_config_and_rename_list(model_class_name, head_name, label2id, num_labels=None):
    if label2id is None:
        logger.warning(
            "No valid map of labels in label2id. Falling back to default (num_labels=2). This may cause errors during loading!"
        )
        label2id = {"LABEL_" + str(i): i for i in range(2)}
    # num_labels is optional (e.g. for regression, when no map given)
    num_labels = num_labels or len(label2id)
    data = STATIC_TO_FLEX_HEAD_MAP[model_class_name]
    # config
    config = data["config"]
    if config["head_type"] == "multiple_choice":
        config["num_choices"] = num_labels
    else:
        config["num_labels"] = num_labels
    config["label2id"] = label2id
    # rename
    rename_list = []
    i = 0
    for name in data["layers"]:
        escaped_name = re.escape(name)
        rename_list.append((rf"{escaped_name}\.(\S+)", f"heads.{head_name}.{i+1}.{{0}}"))
        i += 3 if config["activation_function"] else 2  # there's always a dropout layer in between
    rename_func = lambda k, rename_list=rename_list: _regex_list_rename_func(k, rename_list)

    return config, rename_func
