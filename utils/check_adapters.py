import transformers
from check_repo import get_models
from transformers import ModelAdaptersMixin


MODULES_WITH_ADAPTERS = [
    "modeling_bert",
    "modeling_roberta",
    "modeling_xlm_roberta",
    "modeling_distilbert",
]


def check_models_implement_mixin():
    """Checks that all model classes belonging to modules that have adapter-support implemented properly derive the adapter mixin."""
    failures = []
    for module in MODULES_WITH_ADAPTERS:
        for model_name, model_class in get_models(getattr(transformers, module)):
            if not issubclass(model_class, ModelAdaptersMixin):
                failures.append(f"{model_name} should implement ModelAdaptersMixin.")
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    print("Checking all adapter-supporting models implement mixin.")
    check_models_implement_mixin()
