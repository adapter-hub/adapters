
MODEL_NAME = "facebook/bart-base"

from datasets import load_dataset


dataset = load_dataset("web_nlg", "release_v2.1", split="train[:100]")

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_batch(batch):
    full_src_lst = []
    full_tgt_lst = []

    for lexs, triples in zip(batch['lex'], batch['modified_triple_sets']):
        flat_triples = [item for sublist in triples["mtriple_set"] for item in sublist]
        # concat all triples with "&"
        temp_triples = "&".join(flat_triples)

        for i in range(len(lexs["lid"])):
            if lexs["comment"][i] == 'good':
                full_tgt_lst.append(lexs["text"][i])
                full_src_lst.append(temp_triples)

    batch = tokenizer(full_src_lst)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(full_tgt_lst)
    batch["labels"] = labels["input_ids"]

    return batch


# Encode the input data
dataset = dataset.map(encode_batch, batched=True, remove_columns=dataset.column_names)

# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from transformers import AutoModelForSeq2SeqLM, AutoModelWithHeads, PrefixTuningConfig


model = AutoModelWithHeads.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)



# Add a new adapter
config = PrefixTuningConfig(prefix_length=10)
model.add_adapter("web_nlg", config=config)
# Add a matching classification head
model.add_seq2seq_lm_head("web_nlg")
# Activate the adapter
model.train_adapter("web_nlg")


from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq


loader = DataLoader(dataset, batch_size=10, collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model))

for batch in loader:
    print(model(**batch))
    break

def generate(text):
   model.eval()
   input_ids = tokenizer.encode(text, 
                               return_tensors="pt")
   outputs = model.generate(input_ids)
   return tokenizer.decode(outputs[0])

print(generate("Russia | leader | Putin & Russia | capital | Moscow"))
