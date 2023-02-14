# Adapter Activation and Composition

One of the great advantages of using adapters is the possibility to combine multiple adapters trained on different tasks in various ways.
To enable such adapter compositions, `adapter-transformers` comes with a modular and flexible concept to define how the input to the model should flow through the available adapters.
This not only allows stacking ([_MAD-X_](https://arxiv.org/pdf/2005.00052.pdf)) and fusing ([_AdapterFusion_](https://arxiv.org/pdf/2005.00247.pdf)) adapters, but also even more complex adapter setups.

## Adapter Activation

The single location where all the adapter composition magic happens is the `active_adapters` property of the model class.
In the simplest case, you can set the name of a single adapter here to activate it:
```python
model.active_adapters = "adapter_name"
```

Note that we also could have used `model.set_active_adapters("adapter_name")` which does the same.

```{eval-rst}
.. important::
    ``active_adapters`` defines which of the available adapters are used in each forward and backward pass through the model. This means:

    - You cannot activate an adapter not previously added to the model using either ``add_adapter()`` or ``load_adapter()``.
    - All adapters not mentioned anywhere in the ``active_adapters`` setup are ignored although they might be loaded into the model. Thus, after adding an adapter, make sure to activate it.
```

Alternatively, the [`AdapterSetup`](transformers.AdapterSetup) context manager allows dynamic configuration of activated setups without changing the model state:

```python
model = ...
model.add_adapter("adapter_name")

with AdapterSetup("adapter_name"):
    # will use the adapter named "adapter_name" in the forward pass
    outputs = model(**inputs)
```

## Composition Blocks - Overview

The basic building blocks of the more advanced setups are simple objects derived from `AdapterCompositionBlock`,
each representing a different possibility to combine single adapters.
The following table gives an overview on the supported composition blocks and their support by different adapter methods.

| Block | (Bottleneck)<br> Adapters | Prefix<br> Tuning | Compacter | LoRA | (IA)³ |
| --- | --- | --- | --- | --- | --- |
| [`Stack`](#stack) | ✅ | ✅ | ✅ |  |  |
| [`Fuse`](#fuse) | ✅ |  | ✅ |  |  |
| [`Split`](#split) | ✅ |  | ✅ |  |  |
| [`BatchSplit`](#batchsplit) | ✅ | ✅ | ✅ |  |  |
| [`Parallel`](#parallel) | ✅ | ✅ | ✅ |  |  |

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
For more, check out [this Colab notebook](https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/04_Cross_Lingual_Transfer.ipynb) on cross-lingual transfer.

In the following example, we stack the adapters `a`, `b` and `c` so that in each layer, the input is first passed through `a`, the output of `a` is then inputted to `b` and the output of `b` is finally inputted to `c`.

```python
import transformers.adapters.composition as ac

// ...

model.add_adapter("a")
model.add_adapter("b")
model.add_adapter("c")

model.active_adapters = ac.Stack("a", "b", "c")
```

Since v3.2.0, stacking is also supported for prefix tuning.
Stacked prefixes are prepended to the input states from right to left, i.e. `Stack("a", "b", "c")` will first prepend prefix states for "a" to the input vectors, then prepend "b" to the resulting vectors etc.

In v1.x of `adapter-transformers`, stacking adapters was done using a list of adapter names, i.e. the example from above would be defined as `["a", "b", "c"]`.
For backwards compatibility, you can still do this, although it is recommended to use the new syntax.

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
The fusion layer is added beforehand using `model.add_adapter_fusion()` where we specify the names of the adapters which should be fused.

```python
import transformers.adapters.composition as ac

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

To learn how training an _AdapterFusion_ layer works, check out [this Colab notebook](https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/03_Adapter_Fusion.ipynb) from the `adapter-transformers` repo.

In v1.x of `adapter-transformers`, fusing adapters was done using a nested list of adapter names, i.e. the example from above would be defined as `[["d", "e", "f"]]`.
For backwards compatibility, you can still do this, although it is recommended to use the new syntax.

#### Retrieving AdapterFusion attentions

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

    Splitting the input between two adapters using the 'Stack' block.
```

The `Split` block can be used to split an input sequence between two adapters.
This is done by specifying a split index, at which the sequences should be divided.
In the following example, we split each input sequence between adapters `g` and `h`.
For each sequence, all tokens from 0 up to 63 are forwarded through `g` while all tokens beginning at index 64 are forwarded through `h`:

```python
import transformers.adapters.composition as ac

// ...

model.add_adapter("g")
model.add_adapter("h")

model.active_adapters = ac.Split("g", "h", split_index=64)
```

## `BatchSplit`
The `BatchSplit` lock is an alternative to split the input between several adapters. It does not split the input sequences but the 
batch into smaller batches. As a result, the input sequences remain untouched. 

In the following example, we split the batch between adapters `i`, `k` and `l`. The `batch_sizes`parameter specifies 
the batch size for each of the adapters. The adapter `i` gets two sequences, `k`gets 1 sequence and `l` gets two sequences.
If all adapters should get the same batch size this can be specified by passing one batch size e.g. `batch_sizes = 2`. The sum
specified batch has to match the batch size of the input.
```python
import transformers.adapters.composition as ac

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

## Nesting composition blocks

Of course, it is also possible to combine different composition blocks in one adapter setup.
E.g., we can nest a `Split` block within a `Stack` of adapters:

```python
import transformers.adapters.composition as ac

model.active_adapters = ac.Stack("a", ac.Split("b", "c", split_index=60))
```

However, combinations of adapter composition blocks cannot be arbitrarily deep. All currently supported possibilities are visualized in the figure below. 

```{eval-rst}
.. figure:: img/adapter_blocks_nesting.png
    :height: 300
    :align: center
    :alt: Adapter composition block combinations

    Allowed nestings of adapter composition blocks.
```
