import argparse

from backwards_compatibility.package_comparison.Utils import compare_to_ref_output, create_ref_outputs


# you first need to create a new virtual environment, on which the old package is installed; you can use the requirements_old.txt file
# specify the path to the python executable of the virtual environment with the old adapter-transformers package
argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str)
argparser.add_argument("--model", type=str)
args = argparser.parse_args()
python_path_back_comp_venv = args.path
model = args.model

create_ref_outputs(
    venv_python_path=python_path_back_comp_venv, file_path="create_reference_outputs.py", model_name=model
)

compare_to_ref_output(model_name=model, rtol=1e-05, atol=1e-05)
