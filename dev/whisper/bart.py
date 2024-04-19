from adapters import BartAdapterModel

model = BartAdapterModel.from_pretrained("facebook/bart-base")

model.add_classification_head("bart_head", num_labels=2)