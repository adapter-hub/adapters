# TODO: Replace this with a proper colab notebook
import torch

import adapters
from transformers import AutoModelForSequenceClassification, AutoTokenizer


if __name__ == "__main__":
    """A temporary example to highlight changes implemented for AdapterDrop at inference"""
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    # Convert the model into an adapter model
    adapters.init(model)
    model.load_adapter("sentiment/sst-2@ukp")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize("AdapterHub is awesome!")
    input_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    model.set_active_adapters("sst-2")
    outputs_nodrop = model(input_tensor)

    model.set_active_adapters("sst-2", skip_layers=[0, 1])
    outputs_adapterdrop = model(input_tensor)

    # different probs
    assert not torch.equal(outputs_nodrop[0], outputs_adapterdrop[0])
    # but they should still result in the same prediction
    assert torch.equal(torch.argmax(outputs_nodrop[0]), torch.argmax(outputs_adapterdrop[0]))
