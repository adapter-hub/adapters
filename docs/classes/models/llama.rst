LLaMA
-----------------------------------------------------------------------------------------------------------------------

The LLaMA model was proposed in `LLaMA: Open and Efficient Foundation Language Models <https://arxiv.org/abs/2302.13971>`__ by 
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, 
Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. It is a collection of foundation language 
models ranging from 7B to 65B parameters.

The abstract from the paper is the following:

*We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, 
and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary 
and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, 
Chinchilla-70B and PaLM-540B. We release all our models to the research community.*

LlamaAdapterModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.LlamaAdapterModel
    :members:
    :inherited-members: LlamaPreTrainedModel
