from typing import TYPE_CHECKING

from ....utils import _LazyModule


_import_structure = {
    "adapter_model": [
        "WhisperAdapterModel",
        "WhisperModelWithHeads",
    ],
}


if TYPE_CHECKING:
    from .adapter_model import WhisperAdapterModel, WhisperModelWithHeads

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
