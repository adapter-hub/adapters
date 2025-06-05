from typing import Optional

import torch

from transformers.models.beit.modeling_beit import BeitModel, BeitPreTrainedModel

from ...context import AdapterSetup, ForwardContext
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...utils import inherit_doc_for_adapter_model, inherit_doc_for_function
from ...wrappers import init


@inherit_doc_for_adapter_model(
    model=BeitModel,
    custom_intro="""Beit Model transformer with the option to add multiple flexible heads on top.""",
)
class BeitAdapterModel(ModelWithFlexibleHeadsAdaptersMixin, BeitPreTrainedModel):
    head_types = [
        "image_classification",
    ]
    use_pooler = True

    def __init__(self, config):
        super().__init__(config)

        self.beit = BeitModel(config)
        init(self.beit)

        self._init_head_modules()

        # Initialize weights and apply final processing
        self.post_init()

    # Overwrites the function from: transformers.modeling_utils.PreTrainedModel
    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings specifically for BEiT's tuple output format.
        """

        def make_inputs_require_grads(module, input, output):
            # >>> START AH Changes <<<
            # Handle BEiT's specific tuple output format. Hugging Face's implementation is buggy and doesn't work for BEiT.
            output[0].requires_grad_(True)
            # >>> END AH Changes <<<

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    @inherit_doc_for_function(BeitModel.forward)
    @ForwardContext.wrap
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
                cls_output=pooled_output,  # BEiT does classification based on average-pooling of last hidden state
                head_name=head,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs
