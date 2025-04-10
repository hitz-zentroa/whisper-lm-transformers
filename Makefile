.PHONY: all test style black isort pylint flake8 autopep8 pydocstyle ruff bandit autoflake pydocstringformatter unit doctest kenlm_install install

# Targets to help running the tests.
#
# Example
# -------
# To run the style tests
# ```
# $ make style
# ```
#
# To run the unit tests:
# ```
# $ make unit
# ```
#
# To run all the tests:
# ```
# $ make test
# ```

FILES := $(shell find . -type f -name '*.py')
SRC_FILES := $(shell find src -type f -name '*.py')

# Default target that runs the style, unit and doc tests:
all: test

# Run the style, unit and doc tests:
test: style unit doctest

style: black isort pylint flake8 autopep8 pydocstyle ruff bandit autoflake pydocstringformatter

# Install a recent version of KenLM package from GitHub:
kenlm_install:
	python -m pip install https://github.com/kpu/kenlm/archive/master.zip

# Prepare environment for testing by downloading necessary LMs:
test_req: kenlm_install
	# This replaces the 5gram-$(LLANG).bin target during tests
	@echo "Downloading LM..."
	if [ ! -e 5gram-eu.bin ]; then \
	    wget -O 5gram-eu.bin https://aholab.ehu.eus/~xzuazo/models/Basque%20LMs/5gram.bin ; \
	fi

# Check Python files using black:
black:
	@echo "Checking black..."
	black --check $(FILES)
	@echo

# Sort imports in Python files using isort:
isort:
	@echo "Checking isort..."
	isort  --profile=black --check $(FILES)
	@echo

# Check Python files using pylint:
pylint:
	@echo "Checking pylint..."
	pylint $(FILES)
	@echo

# Check Python files using flake8:
flake8:
	@echo "Checking flake8..."
	flake8 $(FILES)
	@echo

# Auto-format Python files using autopep8:
autopep8:
	@echo "Checking autopep8 for changes..."
	@diffout=$$(autopep8 --diff $(FILES)); \
	if [ -z "$$diffout" ]; then \
	    echo "No changes needed!"; \
	else \
	    echo "Code formatting needed. Here is the diff:"; \
	    echo "$$diffout"; \
	    exit 1; \
	fi
	@echo

# Check Python files for docstring style using pydocstyle:
pydocstyle:
	@echo "Checking pydocstyle..."
	pydocstyle --convention=google $(SRC_FILES)
	@echo

# Check Python files for basic syntax errors using ruff:
ruff:
	@echo "Checking ruff..."
	ruff check $(FILES)
	@echo

# Perform security checks on Python files using bandit:
bandit:
	@echo "Checking bandit..."
	bandit src/
	@echo

# Remove unused imports from Python files using autoflake:
autoflake:
	@echo "Checking autoflake..."
	autoflake $(FILES)
	@echo

# Format Python docstrings using pydocstringformatter:
pydocstringformatter:
	@echo "Checking pydocstringformatter..."
	pydocstringformatter $(FILES)
	@echo

# Run unit tests:
unit: test_req
	@echo "Running pytest..."
	python -m pytest
	@echo

# Test doc examples:
doctest: test_req
	@if [ -z "$(TEST_LLM)" ]; then \
		echo "TEST_LLM environment variable not set. Skipping doctests."; \
	else \
		echo "Running doctest with pytest..."; \
		python -m doctest -v *.md; \
	fi
	@echo

# Install package for production use:
install: kenlm_install
	@echo "Installing package..."
	python -m pip install .
	@echo

# Install package for development use
install-dev: kenlm_install
	@echo "Installing package..."
	python -m pip install -e .[dev]
	@echo
