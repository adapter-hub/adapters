import torch

from transformers import BertModel, RobertaModel, AutoModel
from transformers import AdapterType

from convert_model import load_model_from_old_format


def is_output_equal(model1, model2, adapters=None, iterations=1, input_shape=(1, 128)):
    """Checks whether the output of two models is equal given random input."""
    results = []
    for _ in range(iterations):
        # create some random input
        in_data = torch.randint(1000, input_shape, dtype=torch.long)
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


def run_adapter_test():
    ### A little demo checking the correctness of adapter saving/ loading ###
    global bert_sst

    # load BERT with SST adapter included
    bert_sst = load_model_from_old_format(MODEL_DIR + "sst")
    # load two default BERTs from huggingface
    bert_add_new = BertModel.from_pretrained("bert-base-uncased")

    # save the SST adapter to the file system
    bert_sst.save_adapter(ADAPTER_DIR + "sst", "sst", save_head=True)

    # add SST adapter to BERT by loading the previously saved
    bert_add_new.load_adapter(ADAPTER_DIR + "sst", load_head=True)

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


def run_adapter_download_test():
    global bert_sst

    # load BERT with SST adapter included
    bert_sst = load_model_from_old_format(MODEL_DIR + "sst")

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
    # run_adapter_download_test()
