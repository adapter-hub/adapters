from ....models.xlm_roberta.modeling_xlm_roberta import XLM_ROBERTA_START_DOCSTRING, XLMRobertaConfig
from ....utils import add_start_docstrings
from ..roberta.adapter_model import RobertaAdapterModel, RobertaModelWithHeads


@add_start_docstrings(
    """XLM-RoBERTa Model with the option to add multiple flexible heads on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaAdapterModel(RobertaAdapterModel):
    """
    This class overrides :class:`~transformers.RobertaAdapterModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """XLM-RoBERTa Model with the option to add multiple flexible heads on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModelWithHeads(RobertaModelWithHeads):
    """
    This class overrides :class:`~transformers.RobertaModelWithHeads`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
