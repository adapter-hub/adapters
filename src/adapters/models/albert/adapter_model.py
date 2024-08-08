from transformers.models.albert.modeling_albert import (
    ALBERT_INPUTS_DOCSTRING,
    ALBERT_START_DOCSTRING,
    AlbertModel,
    AlbertPreTrainedModel,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...context import AdapterSetup
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...wrappers import init


@add_start_docstrings(
    """Albert Model transformer with the option to add multiple flexible heads on top.""",
    ALBERT_START_DOCSTRING,
)
class AlbertAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, AlbertPreTrainedModel):
    head_types = [
        "classification",
        "multilabel_classification",
        "tagging",
        "multiple_choice",
        "question_answering",
        "masked_lm",
    ]
    use_pooler = True

    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        init(self.albert)

        self._init_head_modules()

        self.init_weights()

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, context = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            output_context=True,
        )
        # required e.g. for prompt tuning in all models
        kwargs["context"] = context

        # BERT & RoBERTa & ALBERT return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs
