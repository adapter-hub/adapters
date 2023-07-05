# flake8: noqa: E402
import sys
from os.path import abspath, dirname, join


sys.path.insert(1, abspath(join(dirname(dirname(__file__)), "hf_transformers")))

import utils
from utils.check_inits import check_all_inits


utils.check_inits.PATH_TO_TRANSFORMERS = "src/adapters"
check_all_inits()
