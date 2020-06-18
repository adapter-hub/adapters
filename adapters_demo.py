import torch
from pytorch_model_summary import summary

from transformers import (
    BertModel, BertConfig, RobertaModel, AutoModel,
    BertForSequenceClassification, BertModelWithHeads
)
from transformers import AdapterType


def is_output_equal(model1, model2, adapters=None, iterations=1, input_shape=(1, 128)):
    """Checks whether the output of two models is equal given random input."""
    results = []
    for _ in range(iterations):
        # create some random input
        in_data = torch.randint(0, 1000, input_shape, dtype=torch.long)
        if adapters:
            output1 = model1(in_data, adapter_tasks=adapters)[0]
            output2 = model2(in_data, adapter_tasks=adapters)[0]
        else:
            output1 = model1(in_data)[0]
            output2 = model2(in_data)[0]
        results.append(torch.equal(output1, output2))
    return all(results)


def is_model_equal(model1, model2):
    """Checks whether all parameters of two models are equal."""
    results = []
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        # if not torch.equal(p1.data, p2.data):
        #     print(n1, n2)
        results.append(torch.equal(p1.data, p2.data))
    return all(results)


def model_diff(model1, model2):
    params1 = [n for (n, v) in model1.named_parameters()]
    params2 = [n for (n, v) in model2.named_parameters()]
    return [s for s in params1 if s not in params2], [s for s in params2 if s not in params1]


def print_params(model, grads=False):
    for s, param in model.named_parameters():
        if grads:
            print("{:80} requires_grad={}".format(s, param.requires_grad))
        else:
            print(s)


def print_summary(model, input_shape=(1, 128), max_depth=2):
    print(summary(model, torch.zeros(input_shape, dtype=torch.long), max_depth=max_depth))


def run_adapter_test():
    ### A little demo checking the correctness of adapter saving/ loading ###
    global bert_sst

    # load BERT with SST adapter included
    bert_sst = BertModel.from_pretrained(MODEL_DIR + "sst")
    # load two default BERTs from huggingface
    bert_add_new = BertModel.from_pretrained("bert-base-uncased")

    # save the SST adapter to the file system
    bert_sst.save_adapter(ADAPTER_DIR + "sst", "sst")

    # add SST adapter to BERT by loading the previously saved
    bert_add_new.load_adapter(ADAPTER_DIR + "sst")

    # check equality
    assert is_output_equal(bert_add_new, bert_sst, adapters=['sst'])
    assert is_model_equal(bert_add_new, bert_sst)

    print_params(bert_add_new)


def run_lang_adapter_test():
    global roberta

    roberta = RobertaModel.from_pretrained("roberta-base")
    roberta.add_adapter("dummy", "text_lang")
    roberta.save_adapter(ADAPTER_DIR + "dummy", "dummy")

    roberta2 = RobertaModel.from_pretrained("roberta-base")
    roberta2.load_adapter(ADAPTER_DIR + "dummy", "text_lang")

    assert is_output_equal(roberta, roberta2)
    assert is_model_equal(roberta, roberta2)

    print_params(roberta2)


def run_flex_head_test():
    global bert_with_head, bert_with_head2

    bert_with_head = BertModelWithHeads.from_pretrained("bert-base-uncased")
    bert_with_head.add_classification_head("dummy", 2)
    bert_with_head.save_head(ADAPTER_DIR + "dummy_head", "dummy")

    bert_with_head2 = BertModelWithHeads.from_pretrained("bert-base-uncased")
    bert_with_head2.load_head(ADAPTER_DIR + "dummy_head")
    bert_with_head2.active_head = "dummy"

    assert is_model_equal(bert_with_head, bert_with_head2)
    assert is_output_equal(bert_with_head, bert_with_head2)

    print_summary(bert_with_head2)


def run_hf_head_test():
    global bert_for_seq_class, bert_for_seq_class2

    bert_for_seq_class = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    bert_for_seq_class.save_head(ADAPTER_DIR + "seq_class_head")

    bert_for_seq_class2 = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    bert_for_seq_class2.load_head(ADAPTER_DIR + "seq_class_head")

    assert is_model_equal(bert_for_seq_class, bert_for_seq_class2)
    assert is_output_equal(bert_for_seq_class, bert_for_seq_class2)

    print_summary(bert_for_seq_class2)


def run_adapter_download_test():
    global bert_sst

    # load BERT with SST adapter included
    bert_sst = BertModel.from_pretrained(MODEL_DIR + "sst")

    # download pretrained adapter
    bert_base = BertModel.from_pretrained("bert-base-uncased")
    bert_base.load_adapter("sst", "text_task", strict=False)

    # check equality
    for k, v in bert_sst.config.adapters.get("sst").items():
        assert bert_base.config.adapters.get("sst")[k] == v
    assert is_output_equal(bert_sst, bert_base, adapters=['sst'])
    assert is_model_equal(bert_sst, bert_base)


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    MODEL_DIR = "../data/Adapters_16_Bert_Base/"
    ADAPTER_DIR = "../data/adapters/"

    # run_adapter_test()
    # run_lang_adapter_test()
    # run_flex_head_test()
    # run_hf_head_test()
    # run_adapter_download_test()
