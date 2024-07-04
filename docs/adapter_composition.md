# Adapter Activation and Composition

With `adapters`, it becomes possible to combine multiple adapters trained on different tasks in so-called *adapter compositions*.
To enable such compositions, `adapters` comes with a modular and flexible concept to define how the input to the model should flow through the available adapters.
This allows, e.g., stacking ([_MAD-X_](https://arxiv.org/pdf/2005.00052.pdf)) and fusing ([_AdapterFusion_](https://arxiv.org/pdf/2005.00247.pdf)) adapters and even more complex adapter setups.

## Adapter Activation

The single location where all the adapter composition magic happens is the `active_adapters` property of the model class.
In the simplest case, you can set the name of a single adapter here to activate it:
```python
model.active_adapters = "adapter_name"
```

```{eval-rst}
.. important::
    ``active_adapters`` defines which available adapters are used in each forward and backward pass through the model. This means:

    - You cannot activate an adapter before previously adding it to the model using either ``add_adapter()`` or ``load_adapter()``.
    - All adapters not mentioned in the ``active_adapters`` setup are ignored, although they might have been loaded into the model. Thus, after adding an adapter, make sure to activate it.
```
Note that we also could have used the `set_active_adapters` method with `model.set_active_adapters("adapter_name")` which does the same.

Alternatively, the [`AdapterSetup`](adapters.AdapterSetup) context manager allows dynamic configuration of activated setups without changing the model state:

```python
from adapters import AdapterSetup

model = ...
model.add_adapter("adapter_name")

with AdapterSetup("adapter_name"):
    # will use the adapter named "adapter_name" in the forward pass
    outputs = model(**inputs)
```

## Composition Blocks - Overview

The basic building blocks of the more advanced setups are objects derived from `AdapterCompositionBlock`,
each representing a different possibility to combine single adapters.
The following table gives an overview on the supported composition blocks and their support by different adapter methods.

| Block                                       | Bottleneck<br> Adapters | Prefix<br> Tuning | Compacter | LoRA | (IA)³ | Prompt Tuning |
| ------------------------------------------- | ----------------------- | ----------------- | --------- | ---- | ----- | ------------- |
| [`Stack`](#stack)                           | ✅                       | ✅                 | ✅         | ✅(*) | ✅(*)  |               |
| [`Fuse`](#fuse)                             | ✅                       |                   | ✅         |      |       |               |
| [`Split`](#split)                           | ✅                       |                   | ✅         |      |       |               |
| [`BatchSplit`](#batchsplit)                 | ✅                       | ✅                 | ✅         | ✅(*) | ✅(*)  |               |
| [`Parallel`](#parallel)                     | ✅                       | ✅                 | ✅         | ✅(*) | ✅(*)  |               |
| [Output averaging](#output-averaging)       | ✅                       |                   | ✅         | ✅(*) | ✅(*)  |               |
| [Parameter averaging](#parameter-averaging) | ✅                       | ✅                 | ✅         | ✅    | ✅     | ✅             |

(*) except for Deberta and GPT-2.

Next, we present all composition blocks in more detail.

## `Stack`

```{eval-rst}
.. figure:: img/stacking_adapters.png
    :height: 300
    :align: center
    :alt: Illustration of stacking adapters.

    Stacking adapters using the 'Stack' block.
```

The `Stack` block can be used to stack multiple adapters on top of each other.
This kind of adapter composition is used e.g. in the _MAD-X_ framework for cross-lingual transfer [(Pfeiffer et al., 2020)](https://arxiv.org/pdf/2005.00052.pdf), where language and task adapters are stacked on top of each other.
For more, check out [this Colab notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/04_Cross_Lingual_Transfer.ipynb) on cross-lingual transfer.

In the following example, we stack the adapters `a`, `b` and `c` so that in each layer, the input is first passed through `a`, the output of `a` is then inputted to `b` and the output of `b` is finally inputted to `c`.

```python
import adapters.composition as ac

// ...

model.add_adapter("a")
model.add_adapter("b")
model.add_adapter("c")

model.active_adapters = ac.Stack("a", "b", "c")
```

```{eval-rst}
.. note::
    When using stacking for prefix tuning the stacked prefixed are prepended to the input states from right to left, i.e. `Stack("a", "b", "c")` will first prepend prefix states for "a" to the input vectors, then prepend "b" to the resulting vectors etc.
```

## `Fuse`

```{eval-rst}
.. figure:: img/Fusion.png
    :height: 300
    :align: center
    :alt: Illustration of AdapterFusion.

    Fusing adapters with AdapterFusion.
```

The `Fuse` block can be used to activate a fusion layer of adapters.
_AdapterFusion_ is a non-destructive way to combine the knowledge of multiple pre-trained adapters on a new downstream task, proposed by [Pfeiffer et al., 2021](https://arxiv.org/pdf/2005.00247.pdf).
In the following example, we activate the adapters `d`, `e` and `f` as well as the fusion layer that combines the outputs of all three.
The fusion layer is added beforehand using `model.add_adapter_fusion()`, where we specify the names of the adapters which should be fused.

```python
import adapters.composition as ac

// ...

model.add_adapter("d")
model.add_adapter("e")
model.add_adapter("f")
model.add_adapter_fusion(["d", "e", "f"])

model.active_adapters = ac.Fuse("d", "e", "f")
```

```{eval-rst}
.. important::
    Fusing adapters with the ``Fuse`` block only works successfully if an adapter fusion layer combining all of the adapters listed in the ``Fuse`` has been added to the model.
    This can be done either using ``add_adapter_fusion()`` or ``load_adapter_fusion()``.
```

To learn how training an _AdapterFusion_ layer works, check out [this Colab notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/03_Adapter_Fusion.ipynb) from the `adapters` repo.

### Retrieving AdapterFusion attentions

Finally, it is possible to retrieve the attention scores computed by each fusion layer in a forward pass of the model.
These scores can be used for analyzing the fused adapter blocks and can serve as the basis for visualizations similar to those in the AdapterFusion paper.
You can collect the fusion attention scores by passing `output_adapter_fusion_attentions=True` to the model forward call.
The scores for each layer will then be saved in the `adapter_fusion_attentions` attribute of the output:

```python
outputs = model(**inputs, output_adapter_fusion_attentions=True)
attention_scores = outputs.adapter_fusion_attentions
```
Note that this parameter is only available to base model classes and [AdapterModel classes](prediction_heads.md#adaptermodel-classes).
In the example, `attention_scores` holds a dictionary of the following form:
```
{
    '<fusion_name>': {
        <layer_id>: {
            '<module_location>': np.array([...]),
            ...
        },
        ...
    },
    ...
}
```

## `Split`

```{eval-rst}
.. figure:: img/splitting_adapters.png
    :height: 300
    :align: center
    :alt: Illustration of splitting adapters.

    Splitting the input between two adapters using the 'Split' block.
```

The `Split` block can be used to split an input sequence between multiple adapters.
This is done by specifying split indices at which the sequences should be divided.
In the following example, we split each input sequence between adapters `g` and `h`.
For each sequence, all tokens from 0 up to 63 are forwarded through `g` while the next 64 tokens are forwarded through `h`:

```python
import adapters.composition as ac

// ...

model.add_adapter("g")
model.add_adapter("h")

model.active_adapters = ac.Split("g", "h", splits=[64, 64])
```

## `BatchSplit`

The `BatchSplit` block is an alternative to split the input between several adapters. It does not split the input sequences but the 
batch into smaller batches. As a result, the input sequences remain untouched. 

In the following example, we split the batch between adapters `i`, `k` and `l`. The `batch_sizes`parameter specifies 
the batch size for each of the adapters. The adapter `i` gets two sequences, `k`gets 1 sequence and `l` gets two sequences.
If all adapters should get the same batch size this can be specified by passing one batch size e.g. `batch_sizes = 2`. The sum
specified batch has to match the batch size of the input.
```python
import adapters.composition as ac

// ...

model.add_adapter("i")
model.add_adapter("k")
model.add_adapter("l")

model.active_adapters = ac.BatchSplit("i", "k", "l", batch_sizes=[2, 1, 2])

```

## `Parallel`

```{eval-rst}
.. figure:: img/parallel.png
    :height: 300
    :align: center
    :alt: Illustration of parallel adapter forward pass.

    Parallel adapter forward pass as implemented by the 'Parallel' block. The input is replicated at the first layer with parallel adapters.
```

The `Parallel` block can be used to enable parallel multi-task training and inference on different adapters, each with their own prediction head.
Parallel adapter inference was first used in _AdapterDrop: On the Efficiency of Adapters in Transformers_ [(Rücklé et al., 2020)](https://arxiv.org/pdf/2010.11918.pdf).

In the following example, we load two adapters for semantic textual similarity (STS) from the Hub, one trained on the STS benchmark, the other trained on the MRPC dataset.
We activate a parallel setup where the input is passed through both adapters and their respective prediction heads.

```python
import adapters.composition as ac

model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

adapter1 = model.load_adapter("sts/sts-b@ukp")
adapter2 = model.load_adapter("sts/mrpc@ukp")

model.active_adapters = ac.Parallel(adapter1, adapter2)

input_ids = tokenizer("Adapters are great!", "Adapters are awesome!", return_tensors="pt")

output1, output2 = model(**input_ids)

print("STS-B adapter output:", output1[0].item())
print("MRPC adapter output:", bool(torch.argmax(output2[0]).item()))
```

## Averaging Outputs or Parameters

Following approaches of ensembling full models at inference time for better generalization, recent work on adapters has explored methods of averaging pre-trained adapters.
This includes averaging output representations of adapters ([Wang et al., 2021](https://arxiv.org/pdf/2109.04877.pdf)) as well as averaging adapter parameters ([Wang et al., 2022](https://arxiv.org/pdf/2205.12410.pdf), [Chronopoulou et al., 2023](https://aclanthology.org/2023.findings-eacl.153.pdf)).
_Adapters_ provides built-in support for both types of inference time averaging methods.

### Output averaging

Output averaging allows to dynamically aggregate the output representations of multiple adapters in a model forward pass via weighted averaging.
This is realized via the `Average` composition block that works similar to other composition blocks.
In the example below, the three adapters are averaged with the weights `0.1` for `m`, `0.6` for `n` and `0.3` for `o`.

```python
import adapters.composition as ac

// ...

model.add_adapter("m")
model.add_adapter("n")
model.add_adapter("o")

model.active_adapters = ac.Average("m", "n", "o", weights=[0.1, 0.6, 0.3])
```

### Merging Adapters
We can create new adapters by combining the parameters of multiple trained adapters, i.e. merging multiple existing adapters into a new one. The `average_adapter()` method provides this functionality:

```python
model.add_adapter("bottleneck_1", "seq_bn")
model.add_adapter("bottleneck_2", "seq_bn")
model.add_adapter("bottleneck_3", "seq_bn")

model.average_adapter(adapter_name="avg", adapter_list=["bottleneck_1", "bottleneck_2", "bottleneck_3"], weights=[-1, 1.2, 0.8])
```
In this example, the parameters of the three added bottleneck adapters are merged (with weights `-1`, `1.2` and `0.8`, respectively) to create a new adapter `avg`.
Note that for this to succeed, all averaged adapters must use the same adapter configuration. Compared to output averaging, parameter averaging of adapters has the advantage of not inducing any additional inference time relative to using a single adapter.

All [adapter methods](https://docs.adapterhub.ml/overview.html#table-of-adapter-methods) support linear merging. In linear merging, the weights of the trained adapters are linearly combined: Let us have *n* adapters and let $\Phi_i$ be all the parameters of adapter *i*, and $\lambda_i$ be the corresponding weight. The merged adapter parameters $\Phi_{merged}$ are calculated as:

$$
\Phi_{merged} = \sum_{i=0}^{N} \lambda_i \Phi_i
$$

The `average_adapter` method only merges the weights of the adapters but does not create a new head. To average the weights of heads, use the `average_head` method.

#### Merging LoRA Adapters
LoRA introduces $A$ and $B$ matrixes with $\Delta W = BA$. Since the B and A matrices are strongly dependent on each other, there are several ways to merge the weights of LoRA adapters. You can choose the combination method by passing the `combine_strategy` parameter to the `average_adapter` method:

1. `combine_strategy = "linear"`: Linear Combination (default). This has been proposed for LoRA by [Chronopoulou et al. (2023)](https://arxiv.org/abs/2311.09344). With $\Phi = \{A, B\}$:
    
    $$
    \Phi_{merged} = \sum_{i=0}^{N} \lambda_i \Phi_i
    $$

2. `combine_strategy = "lora_linear_only_negate_b"` Following [Zhang et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html), this method only uses negative weights for the B-matrix if the weight is negative:

    $$
    A_{merged} &= \sum_{i=0}^{N} |\lambda_i| A_i\\
    B_{merged} &= \sum_{i=0}^{N} \lambda_i B_i
    $$

3. `combine_strategy = "lora_delta_w_svd"`: This method merges the $\Delta W_i$ of each adapter and then performs a singular value decomposition (SVD) to obtain the *A* and *B* LoRA matrices:
    1. For every adapter *i* we calculate: $\Delta W_i = B_i \cdot A_i$
    2. $\Delta W_{new} = \sum_{i=0}^N \lambda_i \cdot W_i$ 
    3. Perform SVD on $\text{SVD}(\Delta W_{new})$ to obtain $A_{new}$ and $B_{new}$

`lora_delta_w_svd` is not supported by Deberta and GPT-2. Example usage of these LoRA-specific merging strategies:

```python
model.add_adapter("lora_1", "seq_bn")
model.add_adapter("lora_2", "seq_bn")
model.add_adapter("lora_3", "seq_bn")

model.average_adapter(
    adapter_name="lora_avg",
    adapter_list=["lora_1", "lora_2", "lora_3"],
    weights=[1, -1, 1],
    combine_strategy="lora_delta_w_svd",
    svd_rank=8
)
# Note that "lora_delta_w_svd" requires the "svd_rank" parameter, which determines the r (rank) of the resulting LoRA adapter after SVD
```

For both output and parameter averaging, passed weights are normalized by default. To disable normalization, pass `normalize_weights=False`.
For more detailed examples and explanations, refer to our [Task Arithmetic notebook](https://github.com/adapter-hub/adapters/tree/main/notebooks/task_arithmetics_in_adapter.ipynb).


```{eval-rst}
.. tip::
    Adding more adapter merging methods is easy: You have to simply modify the ``average_adapter`` method. Most adapter-methods use the default implementation that only supports linear merging in `model_mixin.py <https://github.com/adapter-hub/adapters/blob/main/src/adapters/model_mixin.py>`_. Others like LoRA overwrite this method to add new merging methods like "lora_delta_w_svd", have a look at `lora.py <https://github.com/adapter-hub/adapters/blob/main/src/adapters/methods/lora.py>`_.
```


## Nesting composition blocks

Of course, it is also possible to combine different composition blocks in one adapter setup.
E.g., we can nest a `Split` block within a `Stack` of adapters:

```python
import adapters.composition as ac

model.active_adapters = ac.Stack("a", ac.Split("b", "c", splits=60))
```

However, combinations of adapter composition blocks cannot be arbitrarily deep. All currently supported possibilities are visualized in the table below.

| Block                          | Supported Nesting                                 |
| ------------------------------ | ------------------------------------------------- |
| [`Stack`](#stack)              | [str, Fuse, Split, Parallel, BatchSplit, Average] |
| [`Fuse`](#fuse)                | [str, Stack]                                      |
| [`Split`](#split)              | [str, Split, Stack, BatchSplit, Average]          |
| [`Parallel`](#parallel)        | [str, Stack, BatchSplit, Average]                 |
| [`BatchSplit`](#batchsplit)    | [str, Stack, Split, BatchSplit, Average]          |
| [`Average`](#output-averaging) | [str, Stack, Split, BatchSplit]                   |

In the table, `str` represents an adapter, e.g. adapter "a" in the nesting example above. Depending on the individual model, some nested compositions might not be possible.
