# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import doctest
import sys
import warnings
import os
from os.path import abspath, dirname, join


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, git_repo_path)

# add original HF tests folder
sys.path.insert(1, abspath(join(dirname(__file__), "hf_transformers")))

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


# skip pipeline tests of HF by default
os.environ["RUN_PIPELINE_TESTS"] = "false"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "is_pt_tf_cross_test: mark test to run only when PT and TF interactions are tested"
    )
    config.addinivalue_line(
        "markers", "is_pt_flax_cross_test: mark test to run only when PT and FLAX interactions are tested"
    )
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "agent_tests: mark the agent tests that are run on their specific schedule")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)


def pytest_sessionfinish(session, exitstatus):
    # If no tests are collected, pytest exists with code 5, which makes the CI fail.
    if exitstatus == 5:
        session.exitstatus = 0


# Doctest custom flag to ignore output.
IGNORE_RESULT = doctest.register_optionflag('IGNORE_RESULT')

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker


def pytest_collection_modifyitems(items):
    # Exclude the 'test_class' group from the test collection since it's not a real test class and byproduct of the generic test class generation.
    items[:] = [item for item in items if 'test_class' not in item.nodeid]