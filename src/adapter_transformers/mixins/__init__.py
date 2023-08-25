from .bart import BartDecoderAdaptersMixin, BartEncoderAdaptersMixin, BartModelAdaptersMixin
from .bert import BertLayerAdaptersMixin, BertModelAdaptersMixin


# IMPORTANT: Only add classes to this mapping that are not copied into the adapter-transformers package
MODEL_MIXIN_MAPPING = {
    "BartEncoder": BartEncoderAdaptersMixin,
    "BartDecoder": BartDecoderAdaptersMixin,
    "BartModel": BartModelAdaptersMixin,
    "BertLayer": BertLayerAdaptersMixin,
    "BertModel": BertModelAdaptersMixin,
    "RobertaLayer": BertLayerAdaptersMixin,
    "RobertaModel": BertModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
}
