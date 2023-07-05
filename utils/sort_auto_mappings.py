# flake8: noqa: E402
import argparse
import sys
from os.path import abspath, dirname, join


sys.path.insert(1, abspath(join(dirname(dirname(__file__)), "hf_transformers")))

import utils
from utils.sort_auto_mappings import sort_all_auto_mappings


utils.sort_auto_mappings.PATH_TO_AUTO_MODULE = "src/adapters/models/auto"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_only", action="store_true", help="Whether to only check or fix style.")
    args = parser.parse_args()

    sort_all_auto_mappings(not args.check_only)
