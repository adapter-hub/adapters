import torch.nn as nn

from transformers.generation import GenerationMixin
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel

from ...context import ForwardContext
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...utils import inherit_doc_for_adapter_model, inherit_doc_for_function
from ...wrappers import init


@inherit_doc_for_adapter_model(
    model=DistilBertModel,
    custom_intro="""DistilBert Model transformer with the option to add multiple flexible heads on top.""",
)
class DistilBertAdapterModel(
    EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, DistilBertPreTrainedModel, GenerationMixin
):
    head_types = [
        "classification",
        "multilabel_classification",
        "tagging",
        "question_answering",
        "multiple_choice",
        "dependency_parsing",
        "masked_lm",
        "causal_lm",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        init(self.distilbert)

        self._init_head_modules()

        self.init_weights()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if :obj:`new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (:obj:`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @inherit_doc_for_function(DistilBertModel.forward)
    @ForwardContext.wrap
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs = self.forward_head(
            distilbert_output, head_name=head, attention_mask=attention_mask, return_dict=return_dict, **kwargs
        )

        return outputs

    # Copied from RobertaForCausalLM
    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "adapter_input_parallelized": model_kwargs.pop("adapter_input_parallelized", False),
        }
