from .bottleneck import init_bottleneck
from .lora import init_lora
from .reft import init_reft


METHOD_INIT_MAPPING = {
    "bottleneck": init_bottleneck,
    "lora": init_lora,
    "reft": init_reft,
}
