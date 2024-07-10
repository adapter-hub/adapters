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

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

test-adapter-methods:
	python -m pytest --ignore ./tests/models -n auto --dist=loadfile -s -v ./tests/

test-adapter-models:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/models

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/
