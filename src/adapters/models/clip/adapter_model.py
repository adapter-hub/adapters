from typing import Optional

import torch

from transformers.models.clip.modeling_clip import (
    CLIP_INPUTS_DOCSTRING,
    CLIP_START_DOCSTRING,
    CLIPModel,
    CLIPPreTrainedModel,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...context import AdapterSetup, ForwardContext
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...wrappers import init


@add_start_docstrings(CLIP_START_DOCSTRING)
class CLIPAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, CLIPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.clip = CLIPModel(config)
        init(self.clip)

        self._init_head_modules()

        self.post_init()

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
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
