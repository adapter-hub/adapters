from .lora import init_lora
from .reft import init_reft


METHOD_INIT_MAPPING = {
    "lora": init_lora,
    "reft": init_reft,
}
