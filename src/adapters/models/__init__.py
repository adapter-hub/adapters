from .albert.mixin_albert import AlbertModelAdaptersMixin
from .bart.mixin_bart import (
    BartDecoderAdaptersMixin,
    BartDecoderWrapperAdaptersMixin,
    BartEncoderAdaptersMixin,
    BartModelAdaptersMixin,
)
from .beit.mixin_beit import BeitIntermediateAdaptersMixin, BeitModelAdaptersMixin, BeitOutputAdaptersMixin
from .bert.mixin_bert import BertLayerAdaptersMixin, BertModelAdaptersMixin
from .clip.mixin_clip import (
    CLIPEncoderAdaptersMixin,
    CLIPModelAdaptersMixin,
    CLIPTextModelAdaptersMixin,
    CLIPTextTransformerAdaptersMixin,
    CLIPVisionModelAdaptersMixin,
)
from .distilbert.mixin_distilbert import DistilBertModelAdaptersMixin, DistilBertTransformerAdaptersMixin
from .gptj.mixin_gptj import GPTJMLPAdaptersMixin
from .llama.mixin_llama import LlamaModelAdapterMixin
from .t5.mixin_t5 import T5BlockAdaptersMixin, T5ModelAdaptersMixin, T5ModelAdaptersWithHeadsMixin
from .vit.mixin_vit import ViTIntermediateAdaptersMixin, ViTModelAdaptersMixin


# IMPORTANT: Only add classes to this mapping that are not copied into the adapters package
MODEL_MIXIN_MAPPING = {
    "AlbertModel": AlbertModelAdaptersMixin,
    "BartEncoder": BartEncoderAdaptersMixin,
    "BartDecoder": BartDecoderAdaptersMixin,
    "BartModel": BartModelAdaptersMixin,
    "BartDecoderWrapper": BartDecoderWrapperAdaptersMixin,
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
    "MBartEncoder": BartEncoderAdaptersMixin,
    "MBartDecoder": BartDecoderAdaptersMixin,
    "MBartDecoderWrapper": BartDecoderWrapperAdaptersMixin,
    "MBartModel": BartModelAdaptersMixin,
    "GPTJMLP": GPTJMLPAdaptersMixin,
    "RobertaLayer": BertLayerAdaptersMixin,
    "RobertaModel": BertModelAdaptersMixin,
    "T5Block": T5BlockAdaptersMixin,
    "T5Model": T5ModelAdaptersMixin,
    "T5ForConditionalGeneration": T5ModelAdaptersWithHeadsMixin,
    "T5ForQuestionAnswering": T5ModelAdaptersWithHeadsMixin,
    "T5EncoderModel": T5ModelAdaptersMixin,
    "ViTIntermediate": ViTIntermediateAdaptersMixin,
    "ViTModel": ViTModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
    "DebertaModel": BertModelAdaptersMixin,
    "DebertaLayer": BertLayerAdaptersMixin,
    "DebertaV2Model": BertModelAdaptersMixin,
    "DebertaV2Layer": BertLayerAdaptersMixin,
    "BertGenerationEncoder": BertModelAdaptersMixin,
    "BertGenerationLayer": BertLayerAdaptersMixin,
    "LlamaModel": LlamaModelAdapterMixin,
}
