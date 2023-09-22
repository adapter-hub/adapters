from typing import Optional

import torch
from typing import Optional, Tuple, Union

from transformers.models.hubert.modeling_hubert import (
    HUBERT_START_DOCSTRING,
    HUBERT_INPUTS_DOCSTRING,
    HubertModel,
    HubertPreTrainedModel
)

from transformers.modeling_outputs import BaseModelOutput,CausalLMOutput, SequenceClassifierOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

from ...context import AdapterSetup
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...wrappers import init

_CONFIG_FOR_DOC = "HubertConfig"

@add_start_docstrings(
    """HuBERT Model transformer with the option to add multiple flexible heads on top.""",
    HubertModel,
)
class HubertAdapterModel(ModelWithFlexibleHeadsAdaptersMixin, HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.hubert = HubertModel(config)
        init(self.hubert)

        self._init_head_modules()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

