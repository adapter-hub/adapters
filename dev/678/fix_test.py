import adapters
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="auto"
)

adapters.init(model)
model.add_adapter("my_adapter")
model.train_adapter("my_adapter")

model.to("cuda")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
inputs = inputs.to("cuda")

is_accessed = False

def hook_fn(module, args, output):
    global is_accessed
    is_accessed = True
    return output

adapter = model.get_adapter("my_adapter")
first_layer_module = adapter[0]["output_adapter"]
first_layer_module.register_forward_hook(hook_fn)

outputs = model(**inputs)
print(outputs)

assert is_accessed