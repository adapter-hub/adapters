.PHONY: extra_style_checks quality style test test-adapter-methods test-adapter-models test-examples

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples tests src utils

# this target runs checks on all files

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	python utils/custom_init_isort.py --check_only
	python utils/sort_auto_mappings.py --check_only
	flake8 $(check_dirs)
	python utils/check_inits.py

# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python utils/custom_init_isort.py
	python utils/sort_auto_mappings.py

# this target runs checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
	${MAKE} extra_style_checks

# Library Tests

# run all tests in the library
test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# run tests for the adapter methods
test-adapter-methods:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/

# run tests for the adapter models
test-adapter-models:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_models/

# run the core tests for all models
test-adapter-core:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m core

# run the adapter composition tests for all models
test-adapter-composition:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m composition

# run the head tests for all models
test-adapter-heads:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m heads

# run the embedding teasts for all models
test-adapter-embeddings:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m embeddings

# run the class conversion tests for all models
test-adapter-class_conversion:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m class_conversion

# run the prefix tuning tests for all models
test-adapter-prefix_tuning:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m prefix_tuning

# run the prompt tuning tests for all models
test-adapter-prompt_tuning:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m prompt_tuning

# run the reft tests for all models
test-adapter-reft:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m reft

# run the unipelt tests for all models
test-adapter-unipelt:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m unipelt

# run the compacter tests for all models
test-adapter-compacter:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m compacter

# run the bottleneck tests for all models
test-adapter-bottleneck:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m bottleneck

# run the ia3 tests for all models
test-adapter-ia3:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m ia3

# run the lora tests for all models
test-adapter-lora:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m lora

# run the config union tests for all models
test-adapter-config_union:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m config_union

# Run tests for examples
test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/
