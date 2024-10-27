import functools
import logging
from typing import List, Optional, Union

import torch
from torch import nn

from transformers.utils import ModelOutput

from ..composition import AdapterCompositionBlock, BatchSplit, Parallel, parse_heads_from_composition
from ..context import AdapterSetup, ForwardContext
from ..loading import PredictionHeadLoader
from ..model_mixin import ModelWithHeadsAdaptersMixin
from .base import (
    ClassificationHead,
    ImageClassificationHead,
    MultiHeadOutput,
    MultiLabelClassificationHead,
    MultipleChoiceHead,
    PredictionHead,
    QuestionAnsweringHead,
    TaggingHead,
)
from .dependency_parsing import BiaffineParsingHead
from .language_modeling import BertStyleMaskedLMHead, CausalLMHead, Seq2SeqLMHead


logger = logging.getLogger(__name__)

MODEL_HEAD_MAP = {
    "classification": ClassificationHead,
    "multilabel_classification": MultiLabelClassificationHead,
    "tagging": TaggingHead,
    "multiple_choice": MultipleChoiceHead,
    "question_answering": QuestionAnsweringHead,
    "dependency_parsing": BiaffineParsingHead,
    "masked_lm": BertStyleMaskedLMHead,
    "causal_lm": CausalLMHead,
    "seq2seq_lm": Seq2SeqLMHead,
    "image_classification": ImageClassificationHead,
}


class ModelWithFlexibleHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    """
    Adds flexible prediction heads to a model class. Implemented by the XModelWithHeads classes.
    """

    head_types: list = []
    use_pooler: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_to_flex_head = True
        if not hasattr(self, "custom_heads"):
            self.custom_heads = {}
        self._active_heads = []

    def head_type(head_type_str: str):
        """
        Checks which head type the decorated function belongs to and raises an error if the model does not support the
        head type.
        """

        def decorator(f):
            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):
                if head_type_str in self.head_types:
                    return f(self, *args, **kwargs)
                else:
                    raise ValueError(
                        f"This model of type '{self.config.model_type}' does not support head type '{head_type_str}'."
                    )

            return wrapper

        return decorator

    def _init_head_modules(self):
        # HACK connect adapters_config to base model -> this should move to a better place
        self.adapters_config = self.base_model.adapters_config
        # this dict is _only_ used for saving & reloading the configs and should not be modified otherwise
        if not hasattr(self.config, "prediction_heads"):
            self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        for head_name, config in self.config.prediction_heads.items():
            self.add_prediction_head_from_config(head_name, config)

        self._add_tied_weights_keys()

    # The following methods are required for handling LM heads

    def get_output_embeddings(self) -> Union[nn.Module, List[nn.Module]]:
        # Only gets the output embeddings for the currently active head
        embeddings = []
        if not self._active_heads:
            return None
        for head_name in self._active_heads:
            if head_name in self.heads:
                head = self.heads[head_name]
                output_embeddings = head.get_output_embeddings()
                embeddings.append(output_embeddings)

        if len(embeddings) == 1:
            return embeddings[0]
        elif len(embeddings) == 0 or all([e is None for e in embeddings]):
            return None
        else:
            return embeddings

    def set_output_embeddings(self, new_embeddings: Union[nn.Module, List[nn.Module]]):
        # Only sets the output embeddings for the currently active head
        if not self._active_heads:
            return
        if not isinstance(new_embeddings, list):
            new_embeddings = [new_embeddings] * len(self._active_heads)
        for head_name, emb in zip(self._active_heads, new_embeddings):
            if head_name in self.heads:
                head = self.heads[head_name]
                head.set_output_embeddings(emb)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        for head_name, head in self.heads.items():
            output_embeddings = head.get_output_embeddings()
            if output_embeddings is not None and self.config.tie_word_embeddings:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        super().tie_weights()

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if not self.config.tie_word_embeddings:
            for head in self.heads.values():
                old_lm_head = self.get_output_embeddings()
                if old_lm_head is not None:
                    new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
                    self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    # Methods for managing prediction heads

    def add_prediction_head_from_config(
        self,
        head_name: str,
        config: dict,
        overwrite_ok: bool = False,
        set_active: bool = True,
    ):
        head_type = config.pop("head_type")
        # handle cases when id2label, label2id or both are available
        id2label = config.pop("id2label", None)
        if id2label is None:
            label2id = config.pop("label2id", None)
            if label2id is not None:
                id2label = {id_: label for label, id_ in label2id.items()}
        else:
            # don't pass label2id to head_class
            config.pop("label2id", None)
        # re-add id2label map to config
        if id2label is not None:
            config["id2label"] = id2label

        if head_type in self.head_types:
            head_class = MODEL_HEAD_MAP[head_type]
            head = head_class(self, head_name, **config)
            self.add_prediction_head(head, overwrite_ok=overwrite_ok, set_active=set_active)
        elif head_type in self.custom_heads:
            # we have to re-add the head type for custom heads
            self.add_custom_head(head_type, head_name, overwrite_ok=overwrite_ok, **config)
        else:
            raise AttributeError(
                "Given head type '{}' is not known. Please register this head type before loading the model".format(
                    head_type
                )
            )

    def get_prediction_heads_config(self):
        heads = {}
        for head_name, head in self.heads.items():
            heads[head_name] = head.config
        return heads

    def register_custom_head(self, identifier, head):
        self.custom_heads[identifier] = head

    @property
    def active_head(self) -> Union[str, List[str]]:
        """
        The active prediction head configuration of this model. Can be either the name of a single available head
        (string) or a list of multiple available heads. In case of a list of heads, the same base model is forwarded
        through all specified heads.

        Returns:
            Union[str, List[str]]: A string or a list of strings describing the active head configuration.
        """
        if not self._active_heads:
            return None
        elif len(self._active_heads) == 1:
            return self._active_heads[0]
        else:
            return self._active_heads

    @active_head.setter
    def active_head(self, head_name_or_list: Union[str, List[str], AdapterCompositionBlock]):
        if isinstance(head_name_or_list, str):
            if head_name_or_list and head_name_or_list not in self.heads:
                raise ValueError(f"Model does not contain a head with name '{head_name_or_list}'.")
            self._active_heads = [head_name_or_list] if head_name_or_list else None
            # If we set a single head, also switch the label mapping. For multiple head, that doesn't make sense?
            if head_name_or_list:
                self.config.label2id = self.heads[head_name_or_list].config.get("label2id", None)
                self.config.id2label = self.get_labels_dict(head_name_or_list)
        else:
            self._active_heads = head_name_or_list

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. This setting can be overriden by passing
        the `adapter_names` parameter in the `foward()` pass. If no adapter with the given name is found, no module of
        the respective type will be activated. In case the calling model class supports named prediction heads, this
        method will attempt to activate a prediction head with the name of the last adapter in the list of passed
        adapter names.

        Args:
            adapter_setup (list):
                The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        self.base_model.set_active_adapters(adapter_setup, skip_layers)
        # use last adapter name as name of prediction head
        if self.active_adapters:
            head_setup = parse_heads_from_composition(self.active_adapters)
            if isinstance(head_setup, str):
                head_setup = [head_setup]
            if head_setup and all(head in self.heads for head in head_setup):
                self.active_head = head_setup
            else:
                logger.info(
                    "Could not identify valid prediction head(s) from setup '{}'.".format(self.active_adapters)
                )

    def add_custom_head(self, head_type, head_name, overwrite_ok=False, set_active=True, **kwargs):
        if head_type in self.custom_heads:
            head = self.custom_heads[head_type](self, head_name, **kwargs)
            # When a build-in head is added as a custom head it does not have the head_type property
            if not hasattr(head.config, "head_type"):
                head.config["head_type"] = head_type
            self.add_prediction_head(head, overwrite_ok, set_active=set_active)
        else:
            raise AttributeError(
                "The given head as a head_type that is not registered as a custom head yet."
                " Please register the head first."
            )

    def add_prediction_head(
        self,
        head: PredictionHead,
        overwrite_ok: bool = False,
        set_active: bool = True,
    ):
        if head.name not in self.heads or overwrite_ok:
            self.heads[head.name] = head
            # add reference to model config to save all head configs too
            self.config.prediction_heads[head.name] = head.config

            # Set a default label2id map if not given
            if "label2id" in head.config.keys() and head.config["label2id"] is None:
                if "num_labels" in head.config.keys():
                    head.config["label2id"] = {"LABEL_" + str(num): num for num in range(head.config["num_labels"])}
                if "num_choices" in head.config.keys():
                    head.config["label2id"] = {"LABEL_" + str(num): num for num in range(head.config["num_choices"])}

            # In case the added head has tied weights, tie them here.
            self.tie_weights()

            logger.info(f"Adding head '{head.name}' with config {head.config}.")
            if set_active:
                self.active_head = head.name

        else:
            raise ValueError(
                f"Model already contains a head with name '{head.name}'. Use overwrite_ok=True to force overwrite."
            )

    @head_type("classification")
    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
        use_pooler=use_pooler,
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
            head = MultiLabelClassificationHead(
                self, head_name, num_labels, layers, activation_function, id2label, use_pooler
            )
        else:
            head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label, use_pooler)
        self.add_prediction_head(head, overwrite_ok)

    @head_type("image_classification")
    def add_image_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
        use_pooler=use_pooler,
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

    @head_type("multiple_choice")
    def add_multiple_choice_head(
        self,
        head_name,
        num_choices=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
        use_pooler=use_pooler,
    ):
        """
        Adds a multiple choice head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_choices (int, optional): Number of choices. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = MultipleChoiceHead(self, head_name, num_choices, layers, activation_function, id2label, use_pooler)
        self.add_prediction_head(head, overwrite_ok)

    @head_type("tagging")
    def add_tagging_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """
        Adds a token classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = TaggingHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    @head_type("question_answering")
    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """
        Adds a question answering head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = QuestionAnsweringHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    @head_type("dependency_parsing")
    def add_dependency_parsing_head(self, head_name, num_labels=2, overwrite_ok=False, id2label=None):
        """
        Adds a biaffine dependency parsing head on top of the model. The parsing head uses the architecture described
        in "Is Supervised Syntactic Parsing Beneficial for Language Understanding? An Empirical Investigation" (Glavaš
        & Vulić, 2021) (https://arxiv.org/pdf/2008.06788.pdf).

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of labels. Defaults to 2.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            id2label (dict, optional): Mapping from label ids to labels. Defaults to None.
        """
        head = BiaffineParsingHead(self, head_name, num_labels, id2label)
        self.add_prediction_head(head, overwrite_ok)

    @head_type("masked_lm")
    def add_masked_lm_head(self, head_name, activation_function="gelu", layers=2, overwrite_ok=False):
        """
        Adds a masked language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            activation_function (str, optional): Activation function. Defaults to 'gelu'.
            layers (int, optional): Number of layers. Defaults to 2.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = BertStyleMaskedLMHead(self, head_name, layers=layers, activation_function=activation_function)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)

    @head_type("causal_lm")
    def add_causal_lm_head(self, head_name, activation_function="gelu", layers=2, overwrite_ok=False):
        """
        Adds a causal language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            activation_function (str, optional): Activation function. Defaults to 'gelu'.
            layers (int, optional): Number of layers. Defaults to 2.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = CausalLMHead(
            self, head_name, layers=layers, activation_function=activation_function, layer_norm=True, bias=True
        )
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)

    @head_type("seq2seq_lm")
    def add_seq2seq_lm_head(
        self,
        head_name,
        layers=1,
        overwrite_ok=False,
    ):
        """
        Adds a sequence-to-sequence language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            layers (int, optional): Number of layers. Defaults to 1.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = Seq2SeqLMHead(self, head_name, layers=layers)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)

    def delete_head(self, head_name: str):
        """
        Deletes the prediction head with the specified name from the model.

        Args:
            head_name (str): The name of the prediction to delete.
        """
        if head_name not in self.config.prediction_heads:
            logger.info("No prediction head '%s' found for deletion. Skipping.", head_name)
            return
        del self.config.prediction_heads[head_name]
        del self.heads[head_name]
        if self.active_head == head_name:
            self.active_head = None

    def _get_used_heads(self, head_name: str = None):
        if head_name:
            used_heads = [head_name]
        # together with context, check if we have heads at all to allow for models without heads
        elif len(self.heads) > 0 and AdapterSetup.get_context_head_setup():
            used_heads = AdapterSetup.get_context_head_setup()
            if isinstance(used_heads, str):
                used_heads = [used_heads]
        elif self._active_heads:
            used_heads = self._active_heads
        else:
            return []

        head_modules = []
        for head in used_heads:
            if head not in self.heads:
                raise ValueError("Unknown head_name '{}'".format(head))
            head_modules.append(self.heads[head])

        return head_modules

    def forward_head(
        self,
        all_outputs,
        head_name=None,
        cls_output=None,
        attention_mask=None,
        return_dict=False,
        context=None,
        **kwargs,
    ):
        """
        The forward pass through a prediction head configuration. There are three ways to specify the used prediction
        head configuration (in order of priority):

            1. If a head_name is passed, the head with the given name is used.
            2. If the forward call is executed within an ``AdapterSetup`` context, the head configuration is read from
               the context.
            3. If the ``active_head`` property is set, the head configuration is read from there.

        Args:
            all_outputs (dict): The outputs of the base model.
            head_name (str, optional): The name of the prediction head to use. If None, the active head is used.
            cls_output (torch.Tensor, optional): The classification output of the model.
            attention_mask (torch.Tensor, optional): The attention mask of the model.
            return_dict (bool): Whether or not to return a ``ModelOutput`` instead of a plain tuple.
            get_cls_from_eos_tokens (bool):
                If set to True, retrieve classifier token representations from the last <eos> token in the sequence.
                Setting to True requires `eos_mask` to be passed as well.
            **kwargs: Additional keyword arguments passed to the forward pass of the head.
        """
        used_head_modules = self._get_used_heads(head_name)
        if len(used_head_modules) == 0:
            logger.debug("No prediction head is used.")
            return all_outputs

        def _get_head_input(outputs, cls_out, batch):
            # TODO-AH check possible edge cases here
            if isinstance(outputs, ModelOutput):
                inputs = {}
                for key, base_output in outputs.items():
                    if torch.is_tensor(base_output):
                        inputs[key] = base_output[batch[0] : batch[-1] + 1]
                inputs = outputs.__class__(**inputs)
            else:
                inputs = tuple()
                for base_output in outputs:
                    inputs = inputs + (base_output[batch],)
            if cls_out is not None:
                cls_input = cls_out[batch]
            else:
                cls_input = None
            return inputs, cls_input

        # Pass invertible adapter if we have one
        if hasattr(self.base_model, "get_invertible_adapter"):
            inv_adapter = self.base_model.get_invertible_adapter()
            if inv_adapter:
                kwargs["invertible_adapter"] = inv_adapter

        # Set prompt tokens length
        if context is not None:
            prompt_tokens_length = context.get("prompt_tokens_length", None)
            if prompt_tokens_length is not None:
                kwargs["prompt_tokens_length"] = prompt_tokens_length

        if isinstance(self.active_head, BatchSplit):
            if sum(self.active_head.batch_sizes) != all_outputs[0].size()[0]:
                raise ValueError(
                    "The specified batch sizes {} do not match the actual batch size {}".format(
                        self.active_head.batch_sizes, all_outputs[0].size()[0]
                    )
                )
            head_outputs = []
            labels = kwargs.pop("labels", None)
            eos_mask = kwargs.pop("eos_mask", None)
            for i, head in enumerate(self.active_head):
                head_module = self.heads[head]
                batch_idx = range(sum(self.active_head.batch_sizes[:i]), sum(self.active_head.batch_sizes[: i + 1]))
                kwargs["labels"] = labels[batch_idx] if labels is not None else None
                kwargs["eos_mask"] = eos_mask[batch_idx] if eos_mask is not None else None
                head_inputs, head_cls_input = _get_head_input(all_outputs, cls_output, batch_idx)
                # head_attention = attention_mask[batch_idx] if attention_mask is not None else None
                head_output = head_module(head_inputs, head_cls_input, attention_mask, return_dict, **kwargs)
                head_outputs.append(head_output)
            combined_loss = (
                sum([out["loss"] for out in head_outputs])
                if all("loss" in out and out["loss"] is not None for out in head_outputs)
                else None
            )
            return_output = MultiHeadOutput(head_outputs=head_outputs, loss=combined_loss)
        elif self.has_parallel_adapters or isinstance(self.active_head, Parallel):
            if len(self.active_head) != self.adapters_config.active_setup.parallel_channels:
                raise ValueError("The number of parallel adapters and the number of active heads must match.")
            orig_batch_size = all_outputs[0].shape[0] // self.adapters_config.active_setup.parallel_channels
            head_outputs = []
            for i, head in enumerate(self.active_head):
                head_module = self.heads[head]
                batch_idx = range(i * orig_batch_size, (i + 1) * orig_batch_size)
                head_inputs, head_cls_input = _get_head_input(all_outputs, cls_output, batch_idx)
                head_output = head_module(head_inputs, head_cls_input, attention_mask, return_dict, **kwargs)
                head_outputs.append(head_output)
            combined_loss = (
                torch.sum(torch.stack([out["loss"] for out in head_outputs]))
                if all("loss" in out and out["loss"] is not None for out in head_outputs)
                else None
            )
            return_output = MultiHeadOutput(head_outputs=head_outputs, loss=combined_loss)
        elif len(used_head_modules) > 1:
            head_outputs = []
            for head_module in used_head_modules:
                head_outputs.append(head_module(all_outputs, cls_output, attention_mask, return_dict, **kwargs))
            return_output = MultiHeadOutput(head_outputs=head_outputs)
        else:
            head_module = used_head_modules[0]
            return_output = head_module(all_outputs, cls_output, attention_mask, return_dict, **kwargs)

        if isinstance(return_output, ModelOutput):
            for attr in ForwardContext.context_attributes:
                if attr not in return_output and attr in all_outputs:
                    return_output[attr] = all_outputs[attr]
        return return_output

    def get_labels_dict(self, head_name=None):
        """
        Returns the id2label dict for the given hea

        Args:
            head_name: (str, optional) the name of the head which labels should be returned. Default is None.
            If the name is None the labels of the active head are returned

        Returns: id2label

        """
        if head_name is None:
            head_name = self.active_head
        if head_name is None:
            raise ValueError("No head name given and no active head in the model")
        if "label2id" in self.heads[head_name].config.keys() and self.heads[head_name].config["label2id"] is not None:
            return {id_: label for label, id_ in self.heads[head_name].config["label2id"].items()}
        else:
            return None

    def get_labels(self, head_name=None):
        """
        Returns the labels the given head is assigning/predictin

        Args:
            head_name: (str, optional) the name of the head which labels should be returned. Default is None.
            If the name is None the labels of the active head are returned

        Returns: labels

        """
        label_dict = self.get_labels_dict(head_name)
        if label_dict is None:
            return None
        else:
            return list(label_dict.values())

    def adapter_to(
        self, name: str, device: Optional[Union[torch.device, str]] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Moves the adapter with the given name to the specified device and data type.

        Args:
            name (str): The name of the adapter to be moved.
            device (torch.device or str, optional): The device on which the adapter should be moved.
            dtype (torch.dtype, optional): The data type to which the adapter should be cast.
        """
        super().adapter_to(name, device, dtype)
        # Move heads to correct device
        if name in self.heads:
            self.heads[name].to(device=device, dtype=dtype)

    # This method is called during model loading in from_pretrained() to apply the state_dict to the model.
    # Override it to inject adapter head logic.
    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        *args,
        **kwargs,
    ):
        # Filter only weights not part of base model
        loader = PredictionHeadLoader(model, error_on_missing=False, convert_to_flex_head=True)
        filter_func = loader.filter_func(None)
        if state_dict is not None:
            head_state_dict = {key: value for key, value in state_dict.items() if filter_func(key)}
        else:
            head_state_dict = None
        head_name = "default"
        head_config, new_head_state_dict = loader.convert_static_to_flex_head(head_state_dict, load_as=head_name)

        if head_config is not None:
            # add head from config
            if head_name in model.heads:
                logger.warning("Overwriting existing head '{}'".format(head_name))

            model.add_prediction_head_from_config(head_name, head_config, overwrite_ok=True)

        if new_head_state_dict is not None:
            # Always ensure base_model_prefix is added, otherwise loading head weights does not work.
            if len(model.base_model_prefix) > 0 and not any(
                s.startswith(model.base_model_prefix) for s in loaded_keys
            ):
                rename_func = lambda x: model.base_model_prefix + "." + x if x not in head_state_dict else x
                state_dict = {rename_func(k): v for k, v in state_dict.items()}
                loaded_keys = [rename_func(k) for k in loaded_keys]

            for k in head_state_dict:
                del state_dict[k]
                loaded_keys.remove(k)
            for k in new_head_state_dict:
                state_dict[k] = new_head_state_dict[k]
                loaded_keys.append(k)

        return super()._load_pretrained_model(
            model,
            state_dict,
            loaded_keys,
            *args,
            **kwargs,
        )
