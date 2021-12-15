import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..heads import (
    BertStyleMaskedLMHead,
    BiaffineParsingHead,
    CausalLMHead,
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    MultipleChoiceHead,
    QuestionAnsweringHead,
    TaggingHead,
)
from ..layer import AdapterLayer
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)


# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class BertSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter", None)


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class BertOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter", None)


class BertModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer


class BertModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """
    Adds flexible heads to a BERT-based model class.
    """

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "tagging": TaggingHead,
        "multiple_choice": MultipleChoiceHead,
        "question_answering": QuestionAnsweringHead,
        "dependency_parsing": BiaffineParsingHead,
        "masked_lm": BertStyleMaskedLMHead,
        "causal_lm": CausalLMHead,
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
        use_pooler=False,
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

    def add_multiple_choice_head(
        self,
        head_name,
        num_choices=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
        use_pooler=False,
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

    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        head = QuestionAnsweringHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

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

    def add_masked_lm_head(self, head_name, activation_function="gelu", overwrite_ok=False):
        """
        Adds a masked language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            activation_function (str, optional): Activation function. Defaults to 'gelu'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = BertStyleMaskedLMHead(self, head_name, activation_function=activation_function)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)

    def add_causal_lm_head(self, head_name, activation_function="gelu", overwrite_ok=False):
        """
        Adds a causal language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            activation_function (str, optional): Activation function. Defaults to 'gelu'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = CausalLMHead(
            self, head_name, layers=2, activation_function=activation_function, layer_norm=True, bias=True
        )
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)
