# # Test for Roberta
# import torch
# import os 
# import sys 
# fpath = os.path.join(os.path.dirname(__file__),'src')
# sys.path.append(fpath)
# print(sys.path)


# # from pathlib import Path
# # from transformers import RobertaTokenizer 

# import src.transformers

# # from transformers.configuration_utils import PretrainedConfig
# # from transformers.tokenization_utils import PreTrainedTokenizer
# # from transformers.utils import TensorType
# # from transformers.utils import is_torch_available


# # from src.transformers.models.roberta import RobertaConfig, RobertaTokenizer
# # from src.transformers.adapters import RobertaModelWithHeads

# from src.transformers import (RobertaTokenizer, 
#                           RobertaConfig, 
#                           RobertaModelWithHeads,
#                           TrainingArguments, 
#                           AdapterTrainer, 
#                           EvalPrediction, 
#                           TextClassificationPipeline)
# import src.transformers.adapters.composition as ac
# from src.transformers.adapters.composition import Fuse
# model_ckpt = "roberta-base"
# tokenizer = RobertaTokenizer.from_pretrained(model_ckpt)
# print("hello world")

#========================================================================================

# Test for Roberta
import torch
import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'src')
sys.path.append(fpath)
print(sys.path)


import src.transformers


from src.transformers import (BigBirdTokenizer, 
                          BigBirdConfig, 
                          BigBirdModelWithHeads,
                          TrainingArguments, 
                          AdapterTrainer, 
                          EvalPrediction, 
                          TextClassificationPipeline)


import src.transformers.adapters.composition as ac
from src.transformers.adapters.composition import Fuse

model_ckpt = "google/bigbird-roberta-base"
tokenizer = BigBirdTokenizer.from_pretrained(model_ckpt)

print("hello world")

