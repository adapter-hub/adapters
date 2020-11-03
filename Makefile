.PHONY: extra_quality_checks quality style fix-copies test test-reduced test-examples docs


check_dirs := examples templates tests src utils

# Check that source code meets quality standards

# NOTE FOR adapter-transformers: The following check is skipped as not all copies implement adapters yet
	# python utils/check_copies.py
extra_quality_checks:
	python utils/check_repo.py

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	${MAKE} extra_quality_checks

# Format source code automatically and check is there are any problems left that need manual fixing

style:
	black $(check_dirs)
	isort $(check_dirs)

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

test-reduced:
	python -m pytest -n auto --dist=loadfile -s -v\
		--ignore-glob='tests/test_tokenization*'\
		--ignore-glob='tests/test_pipelines*'\
		--ignore-glob='tests/test_hf*'\
		--ignore-glob='tests/test_doc*'\
		./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/

# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W"
