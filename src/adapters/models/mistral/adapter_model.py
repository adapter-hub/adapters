import logging

import torch

from transformers.generation import GenerationMixin
from transformers.models.mistral.modeling_mistral import MISTRAL_START_DOCSTRING, MistralModel, MistralPreTrainedModel
from transformers.utils import add_start_docstrings

from ...composition import adjust_tensors_for_parallel
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...wrappers import init


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
The Mistal Model that allows the loading of different heads for different tasks. This enables a flexible use of the
models and adpters. Since this class does classification on the last token, it requires to know the position of the
last token. If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding
token in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same
(take the last value in each row of the batch).
""",
    MISTRAL_START_DOCSTRING,
)
class MistralAdapterModel(
    EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, MistralPreTrainedModel, GenerationMixin
):
    head_types = [
        "classification",
        "multilabel_classification",
        "tagging",
        "question_answering",
        "causal_lm",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        init(self.model)

        self._init_head_modules()

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, context = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            output_context=True,
        )
        # required e.g. for prompt tuning in all models
        kwargs["context"] = context

        batch_size = outputs[0].shape[0]

        if self.config.pad_token_id is None:
            # TODO-AH: this may result in unexpected behavior for classification. Find a better way to do this?
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                (sequence_lengths,) = adjust_tensors_for_parallel(outputs[0], sequence_lengths)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        cls_logits = outputs[0][range(batch_size), sequence_lengths]

        outputs = self.forward_head(
            outputs,
            head_name=head,
            cls_output=cls_logits,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "adapter_input_parallelized": kwargs.pop("adapter_input_parallelized", False),
            }
        )
        return model_inputs
