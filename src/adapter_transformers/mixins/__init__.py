from .albert import AlbertModelAdaptersMixin
from .bart import BartDecoderAdaptersMixin, BartEncoderAdaptersMixin, BartModelAdaptersMixin
from .beit import BeitIntermediateAdaptersMixin, BeitModelAdaptersMixin, BeitOutputAdaptersMixin
from .bert import BertLayerAdaptersMixin, BertModelAdaptersMixin
from .clip import (
    CLIPEncoderAdaptersMixin,
    CLIPModelAdaptersMixin,
    CLIPTextModelAdaptersMixin,
    CLIPTextTransformerAdaptersMixin,
    CLIPVisionModelAdaptersMixin,
)
from .distilbert import DistilBertModelAdaptersMixin, DistilBertTransformerAdaptersMixin
from .gptj import GPTJMLPAdaptersMixin
from .t5 import T5BlockAdaptersMixin, T5ModelAdaptersMixin, T5ModelAdaptersWithHeadsMixin
from .vit import ViTIntermediateAdaptersMixin, ViTModelAdaptersMixin


# IMPORTANT: Only add classes to this mapping that are not copied into the adapter-transformers package
MODEL_MIXIN_MAPPING = {
    "AlbertModel": AlbertModelAdaptersMixin,
    "BartEncoder": BartEncoderAdaptersMixin,
    "BartDecoder": BartDecoderAdaptersMixin,
    "BartModel": BartModelAdaptersMixin,
    "BeitIntermediate": BeitIntermediateAdaptersMixin,
    "BeitOutput": BeitOutputAdaptersMixin,
    "BeitModel": BeitModelAdaptersMixin,
    "BertLayer": BertLayerAdaptersMixin,
    "BertModel": BertModelAdaptersMixin,
    "Transformer": DistilBertTransformerAdaptersMixin,
    "DistilBertModel": DistilBertModelAdaptersMixin,
    "CLIPEncoder": CLIPEncoderAdaptersMixin,
    "CLIPTextTransformer": CLIPTextTransformerAdaptersMixin,
    "CLIPTextModel": CLIPTextModelAdaptersMixin,
    "CLIPVisionModel": CLIPVisionModelAdaptersMixin,
    "CLIPModel": CLIPModelAdaptersMixin,
    "CLIPTextModelWithProjection": CLIPTextModelAdaptersMixin,
    "CLIPVisionModelWithProjection": CLIPVisionModelAdaptersMixin,
    "GPTJMLP": GPTJMLPAdaptersMixin,
    "RobertaLayer": BertLayerAdaptersMixin,
    "RobertaModel": BertModelAdaptersMixin,
    "T5Block": T5BlockAdaptersMixin,
    "T5Model": T5ModelAdaptersMixin,
    "T5ForConditionalGeneration": T5ModelAdaptersWithHeadsMixin,
    "T5EncoderModel": T5ModelAdaptersMixin,
    "ViTIntermediate": ViTIntermediateAdaptersMixin,
    "ViTModel": ViTModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
    "DebertaModel": BertModelAdaptersMixin,
    "DebertaLayer": BertLayerAdaptersMixin,
    "DebertaV2Model": BertModelAdaptersMixin,
    "DebertaV2Layer": BertLayerAdaptersMixin,
}
