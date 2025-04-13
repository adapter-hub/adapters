# Multi Task Methods

## MTL-LoRA

_Configuration class_: [`MTLLoRAConfig`](adapters.MTLLoRAConfig)

"MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning" ([Yang et al., 2024](https://arxiv.org/pdf/2410.09437)). MTL-LoRA enhances LoRA for multi-task learning (MTL) by improving task differentiation and knowledge sharing. It introduces a task-specific low-rank learnable matrix $\Lambda_t$ to better capture task-specific information and utilizes $n$ low-rank up-projection matrices for diverse information-sharing. A weighted averaging mechanism integrates these matrices, allowing adaptive knowledge transfer across tasks. Specifically, the MTL-LoRA output for task $t$ is formulated as:  

$$
h_t = (W + \Delta W_t)x_t = Wx_t + \sum_{i=1}^n\frac{\text{exp}(w_t^i/\tau)B^i}{\sum_{j=1}^n\text{exp}(w_t^{j}/\tau)}\Lambda_t A x_t
$$

where $\tau$ controls the sharpness of weight distribution. 

`MTL-LoRA` is trainable with `MultiTask` composition and a datasets wich contains `task_ids` column (see. [`MultiTask` Composition](adapter_composition.md#multitask)).


_Example_:
```python
from adapters import MTLLoRAConfig
import adapters.composition as ac

config = MTLLoRAConfig(
    r=8,
    alpha=16,
    n_up_projection=3,
)

model.add_adapter("i", config)
model.add_adapter("k", config)
model.add_adapter("l", config)

model.share_parameters(
    adapter_names=["i", "k", "l"],
)

model.active_adapters = ac.MultiTask("i", "k", "l")
```
