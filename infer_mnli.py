import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdapterType, BertForSequenceClassification, InputFeatures

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#model.load_adapter("sentiment/sst-2@ukp")
from transformers import AdapterConfig
config = AdapterConfig.load("pfeiffer")
#model.load_adapter("nli/multinli@ukp", "text_task", config=config)
model.load_adapter("/home/theorist17/projects/adapter/adapters/MNLI/checkpoint-18000/mnli/", "text_task", config=config)

model.eval()

def predict(sentence, sentence2):
    bert_encoding = tokenizer.encode_plus(tokenizer.tokenize(sentence), text_pair=tokenizer.tokenize(sentence2), max_length=128, pad_to_max_length=True)
    # for key, value in bert_encoding.items():
    #     print("{}:\n\t{}".format(key, value))
    print('Decoded!', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(bert_encoding['input_ids'])))
    
    inputs = {k: torch.tensor(bert_encoding[k]).view(1, -1) for k in bert_encoding }
    # for k, v in inputs.items():
    #     inputs[k] = v.to("cuda:0")
    print('inputs', inputs, type(inputs))
    #print('inputs_ids', inputs['input_ids'][0], type(inputs['input_ids'][0]))

    # predict output tensor
    outputs = model(input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                    adapter_names=['mnli'])
    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     encoder_hidden_states=None,
    #     encoder_attention_mask=None,
    #     adapter_names=None,
    # ):

    m = torch.nn.Softmax(dim=1)
    print(m(outputs[0]))
    # retrieve the predicted class label
    if 0 == torch.argmax(outputs[0]).item():
        return 'contradiction'
    elif 1  == torch.argmax(outputs[0]).item():
        return 'entailment'
    else:
        return 'neutral' 

def test(t, t2):
    print(t, '->', t2)
    print(predict(t, t2))
    print()

with torch.no_grad():
    test("The Old One always comforted Ca'daan, except today", "Ca'daan knew the Old One very well.")
    test("yes now you know if if everybody like in August when everybody's on vacation or something we can dress a little more casual or", "August is a black out month for vacations in the company.")
    test('i love you', 'i kill you')
    test('i love you.', 'i give you my love.')
    test("At the other end of Pennsylvania Avenue, people began to line up for a White House tour.", "People formed a line at the end of Pennsylvania Avenue.")
    test("i love you", "i don't care")
    test('i hate you.', 'i wanna hug you.')
    test("'Hello, Ben.'", "I ignored Ben")
    test('no oh no oh well take care', 'Bye for now.')
    test("Ca'daan continued.", "Ca'daan refused to stop.")
    test("Gays and lesbians.", "Hetrosexuals.")
    test("Sun rise.", "It is morning.")
    test("It's high noon.", "It is the midnight.")
    test("Good morning.", "It is evening.")
    test("Good morning.", "It is morning.")