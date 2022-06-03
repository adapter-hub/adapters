from typing import Optional

import torch

from ....models.vit.modeling_vit import VIT_INPUTS_DOCSTRING, VIT_START_DOCSTRING, ViTModel, ViTPreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...context import AdapterSetup
from ...heads import ImageClassificationHead, ModelWithFlexibleHeadsAdaptersMixin


@add_start_docstrings(
    """Bert Model transformer with the option to add multiple flexible heads on top.""",
    VIT_START_DOCSTRING,
)
class ViTAdapterModel(ModelWithFlexibleHeadsAdaptersMixin, ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(config)

        self._init_head_modules()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
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

    head_types = {
        "image_classification": ImageClassificationHead,
    }

    def add_image_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
        use_pooler=False,
    ):
        """
        Adds an image classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        head = ImageClassificationHead(
            self,
            head_name,
            num_labels=num_labels,
            layers=layers,
            activation_function=activation_function,
            multilabel=multilabel,
            id2label=id2label,
            use_pooler=use_pooler,
        )
        self.add_prediction_head(head, overwrite_ok)
