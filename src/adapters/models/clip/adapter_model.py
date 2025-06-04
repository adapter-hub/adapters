from typing import Optional

import torch

from transformers.models.clip.modeling_clip import CLIPModel, CLIPPreTrainedModel

from ...context import AdapterSetup, ForwardContext
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...utils import inherit_doc_for_adapter_model, inherit_doc_for_function
from ...wrappers import init


@inherit_doc_for_adapter_model(
    model=CLIPModel,
    custom_intro="""CLIP Model with the option to add multiple flexible heads on top.""",
)
class CLIPAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, CLIPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.clip = CLIPModel(config)
        init(self.clip)

        self._init_head_modules()

        self.post_init()

    @inherit_doc_for_function(CLIPModel.forward)
    @ForwardContext.wrap
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
        **kwargs,
    ):
        outputs = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                outputs,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs
