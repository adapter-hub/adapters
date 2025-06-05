from typing import Optional

import torch

from transformers.models.vit.modeling_vit import ViTModel, ViTPreTrainedModel

from ...context import AdapterSetup, ForwardContext
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...utils import inherit_doc_for_adapter_model, inherit_doc_for_function
from ...wrappers import init


@inherit_doc_for_adapter_model(
    model=ViTModel,
    custom_intro="""ViT Model transformer with the option to add multiple flexible heads on top.""",
)
class ViTAdapterModel(ModelWithFlexibleHeadsAdaptersMixin, ViTPreTrainedModel):

    head_types = [
        "image_classification",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(config)
        init(self.vit)

        self._init_head_modules()

        # Initialize weights and apply final processing
        self.post_init()

    @inherit_doc_for_function(ViTModel.forward)
    @ForwardContext.wrap
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # BERT & RoBERTa return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                head_name=head,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs
