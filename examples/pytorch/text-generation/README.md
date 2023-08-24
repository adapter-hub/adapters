<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Language generation with Adapters
> **Note:** We have not adapted the following scripts of Hugging Face Transformers:
> - `run_generation_contrastive_search.py`
>
> To avoid confusion we have not included these non-adapted versions in the examples of Adapters.

Based on the script [`run_generation.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py).

Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL, XLNet, CTRL.
A similar script is used for our official demo [Write With Transfomer](https://transformer.huggingface.co), where you
can try out the different models available in the library.

Example usage:

```bash
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
```

This can also be done by using a trained adapter. With the `--adapter_path` argument you can specify an adapter to load 
for language generation.

Example with Adapters:  
```bash
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
    --load_adapter=./tmp/poem
```