from .albert import AlbertModelAdaptersMixin
from .bart import BartDecoderAdaptersMixin, BartEncoderAdaptersMixin, BartModelAdaptersMixin
from .bert import BertLayerAdaptersMixin, BertModelAdaptersMixin
from .t5 import T5BlockAdaptersMixin, T5ModelAdaptersMixin, T5ModelAdaptersWithHeadsMixin


# IMPORTANT: Only add classes to this mapping that are not copied into the adapter-transformers package
MODEL_MIXIN_MAPPING = {
    "AlbertModel": AlbertModelAdaptersMixin,
    "BartEncoder": BartEncoderAdaptersMixin,
    "BartDecoder": BartDecoderAdaptersMixin,
    "BartModel": BartModelAdaptersMixin,
    "BertLayer": BertLayerAdaptersMixin,
    "BertModel": BertModelAdaptersMixin,
    "RobertaLayer": BertLayerAdaptersMixin,
    "RobertaModel": BertModelAdaptersMixin,
    "T5Block": T5BlockAdaptersMixin,
    "T5Model": T5ModelAdaptersMixin,
    "T5ForConditionalGeneration": T5ModelAdaptersWithHeadsMixin,
    "T5EncoderModel": T5ModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
}
