import warnings

import torch

from ....models.whisper.modeling_whisper import (
    WHISPER_INPUTS_DOCSTRING,
    WHISPER_START_DOCSTRING,
    WhisperConfig,
    WhisperModel,
    WhisperPreTrainedModel,
    shift_tokens_right,
)
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...composition import adjust_tensors_for_parallel
from ...heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    QuestionAnsweringHead,
    Seq2SeqLMHead,
)
from ...model_mixin import EmbeddingAdaptersWrapperMixin


@add_start_docstrings(
    "WHISPER Model with the option to add multiple flexible prediction heads on top.", WHISPER_START_DOCSTRING
)
class WhisperAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, WhisperPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"proj_out.weight"]

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = WhisperModel(config)

        self._init_head_modules()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ):
        # TODO What should be done before the original model.forward call?

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
        )
        
        # TODO What should be done after the original model.forward call?
        # TODO Language detection?

        head_outputs = self.forward_head(
            outputs,
            head_name=head,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        return head_outputs

    # Copied from WhisperForConditionalGeneration
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }

    # Copied from WhisperForConditionalGeneration
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "question_answering": QuestionAnsweringHead,
        "seq2seq_lm": Seq2SeqLMHead,
    }

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
    ):
        """
        Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        if multilabel:
            head = MultiLabelClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        else:
            head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    def add_qa_head(
        self,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
    ):
        head = QuestionAnsweringHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    def add_seq2seq_lm_head(
        self,
        head_name,
        overwrite_ok=False,
    ):
        """
        Adds a sequence-to-sequence language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = Seq2SeqLMHead(self, head_name)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)


class WhisperModelWithHeads(WhisperAdapterModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                self.__class__.__bases__[0].__name__
            ),
            FutureWarning,
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
