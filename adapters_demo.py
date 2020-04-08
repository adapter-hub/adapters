import torch

from transformers.modeling_bert import BertModel
from combine_adapters import copy_adapter_weights
from transformers.Adapters import *


def is_output_equal(model1, model2, adapters=None, iterations=1, input_shape=(1, 128)):
    """Checks whether the output of two models is equal given random input."""
    results = []
    if adapters:
        model1.enable_adapters()
        model2.enable_adapters()
    else:
        model1.disable_adapters()
        model2.disable_adapters()
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
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        results.append(torch.equal(param1.data, param2.data))
    return all(results)

def model_diff(model1, model2):
    params1 = [n for (n,v) in model1.named_parameters()]
    params2 = [n for (n,v) in model2.named_parameters()]
    return [s for s in params1 if not s in params2], [s for s in params2 if not s in params1]

def _print_params(model):
    for s, _ in model.named_parameters():
        print(s)

def run_adapter_test():
    ### A little demo checking the correctness of adapter saving/ loading ###
    global bert_sst

    # load BERT with SST adapter included
    bert_sst = BertModel.from_pretrained(MODEL_DIR+"sst")
    # load two default BERTs from huggingface
    bert_add_old = BertModel.from_pretrained("bert-base-uncased")
    bert_add_new = BertModel.from_pretrained("bert-base-uncased")

    # save the SST adapter to the file system
    bert_sst.save_adapter(ADAPTER_DIR+"sst", "sst")

    # add SST adapter to BERT using old method
    bert_add_old.config.adapters = []
    bert_add_old.config.adapter_config = bert_sst.config.adapter_config
    copy_adapter_weights(bert_sst, bert_add_old)

    # add SST adapter to BERT by loading the previously saved
    bert_add_new.load_adapter(ADAPTER_DIR+"sst")

    # check equality
    assert is_output_equal(bert_add_new, bert_sst, adapters=['sst'])
    assert is_output_equal(bert_add_new, bert_add_old, adapters=['sst'])
    assert is_model_equal(bert_add_new, bert_sst)
    assert is_model_equal(bert_add_new, bert_add_old)

    _print_params(bert_add_new)


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    MODEL_DIR = "../data/Adapters_16_Bert_Base/"
    ADAPTER_DIR = "../data/adapters/"

    run_adapter_test()
