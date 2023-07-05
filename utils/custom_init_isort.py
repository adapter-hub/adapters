# flake8: noqa: E402
import argparse
import sys
from os.path import abspath, dirname, join


sys.path.insert(1, abspath(join(dirname(dirname(__file__)), "hf_transformers")))

import utils
from utils.custom_init_isort import sort_imports_in_all_inits


utils.custom_init_isort.PATH_TO_TRANSFORMERS = "src/adapters"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_only", action="store_true", help="Whether to only check or fix style.")
    args = parser.parse_args()

    sort_imports_in_all_inits(check_only=args.check_only)
