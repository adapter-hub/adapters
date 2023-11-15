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
from .gpt2.mixin_gpt2 import GPT2ModelAdapterMixin
from .gptj.mixin_gptj import GPTJMLPAdaptersMixin, GPTJModelAdapterMixin
from .llama.mixin_llama import LlamaModelAdapterMixin
from .t5.mixin_t5 import (
    T5BlockAdaptersMixin,
    T5ForCondiditionalGenerationWithHeadsMixin,
    T5ForQuestionAnsweringWithHeadsMixin,
    T5ModelAdaptersMixin,
)
from .vit.mixin_vit import ViTIntermediateAdaptersMixin, ViTModelAdaptersMixin
from .xmod.mixin_xmod import XmodModelAdaptersMixin


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
    "ElectraLayer": BertLayerAdaptersMixin,
    "ElectraModel": BertModelAdaptersMixin,
    "MBartEncoder": BartEncoderAdaptersMixin,
    "MBartDecoder": BartDecoderAdaptersMixin,
    "MBartDecoderWrapper": BartDecoderWrapperAdaptersMixin,
    "MBartModel": BartModelAdaptersMixin,
    "GPT2Model": GPT2ModelAdapterMixin,
    "GPTJMLP": GPTJMLPAdaptersMixin,
    "GPTJModel": GPTJModelAdapterMixin,
    "RobertaLayer": BertLayerAdaptersMixin,
    "RobertaModel": BertModelAdaptersMixin,
    "T5Block": T5BlockAdaptersMixin,
    "T5Model": T5ModelAdaptersMixin,
    "T5ForConditionalGeneration": T5ForCondiditionalGenerationWithHeadsMixin,
    "T5ForQuestionAnswering": T5ForQuestionAnsweringWithHeadsMixin,
    "T5EncoderModel": T5ModelAdaptersMixin,
    "ViTIntermediate": ViTIntermediateAdaptersMixin,
    "ViTModel": ViTModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
    "XmodLayer": BertLayerAdaptersMixin,
    "XmodModel": XmodModelAdaptersMixin,
    "DebertaModel": BertModelAdaptersMixin,
    "DebertaLayer": BertLayerAdaptersMixin,
    "DebertaV2Model": BertModelAdaptersMixin,
    "DebertaV2Layer": BertLayerAdaptersMixin,
    "BertGenerationEncoder": BertModelAdaptersMixin,
    "BertGenerationLayer": BertLayerAdaptersMixin,
    "LlamaModel": LlamaModelAdapterMixin,
}
