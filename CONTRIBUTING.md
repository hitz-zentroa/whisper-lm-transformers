# Contributing to Whisper-LM-Transformers

Thank you for your interest in contributing to **Whisper-LM-Transformers**!

This document outlines how to set up your development environment, run tests,
and ensure code style consistency.

## Development Environment

1. **Clone the repository**:

    ```bash
    git clone https://github.com/hitz-zentroa/whisper-lm-transformers.git
    cd whisper-lm-transformers
    ```

2. **Install in editable mode** with development dependencies:

    ```bash
    pip install -e .[dev]
    ```

   This installs the package locally in editable mode, plus the extra "dev"
   tools used for linting, style checks, and testing.

3. **(Optional) GPU Support**:

   - Some tests require a GPU for Large Language Models. These tests are
     disabled by default. To enable them, set the environment variable
    `TEST_LLM=1` (e.g., `export TEST_LLM=1`) before running the tests. Without
    a capable GPU, these tests may fail or be extremely slow.

---

## Makefile Commands

We use a **Makefile** to simplify running style/linter checks and tests. Below
are the available targets and what they do:

- **`make install`**: Installs the package in the current environment
  (`pip install .`).
- **`make install-dev`**: Installs the package plus development dependencies
  (`pip install -e .[dev]`).
- **`make style`**: Runs all style and linter checks (see below).
- **`make test`**: Runs both `style` checks and the tests
  (`unit` + `doctest`).
- **`make unit`**: Runs unit tests
  (`pytest tests/`).
- **`make doctest`**: Runs doctests against Markdown (`*.md`) or Python
  docstrings.
- **`make all`**: Alias for `make test`.

Here’s the summary of each style/linter tool that runs under `make style`:

1. [`black`](https://black.readthedocs.io/en/stable/) – "the uncompromising
   code formatter" that ensures consistent formatting.
2. [`isort`](https://timothycrosley.github.io/isort/) – sorts imports
   automatically for consistency.
3. [`pylint`](https://github.com/PyCQA/pylint) – a static code analyzer that
   checks code for errors, style, etc.
4. [`flake8`](https://flake8.pycqa.org/en/latest/) – a wrapper around PyFlakes,
   pycodestyle, and Ned Batchelder’s McCabe script.
6. [`pydocstyle`](http://www.pydocstyle.org/en/stable/) – enforces docstring
   conventions ([PEP 257](https://peps.python.org/pep-0008/)).
7. [`ruff`](https://github.com/charliermarsh/ruff/) – a fast Python
   linter/formatter (written in Rust).
8. [`bandit`](https://github.com/PyCQA/bandit) – security checks on Python
   code.
9. [`autoflake`](https://github.com/myint/autoflake) – removes unused
   imports/variables automatically.
10.[`pydocstringformatter`](https://github.com/DanielNoord/pydocstringformatter)
   – automatically formats docstrings to follow
   [PEP 257](https://peps.python.org/pep-0008/).

The command `make style` runs all these style/linter checks in sequence to
ensure your code meets the repository’s standards.

---

## Running Tests

### Unit Tests

- **Unit tests** reside in the `tests/` folder. We use `pytest` to run them:

```bash
make unit
```

This runs all discovered tests in `tests/`.

If you want to run them manually or pass extra flags, you can do:

```bash
pytest --disable-warnings -v
```

### Doctests

We also use Python’s built-in doctest to verify code snippets in docstrings or
Markdown files. Run them via:

```
make doctest
```

This checks for pieces of text that look like interactive Python sessions
(`>>> …`) and executes them to ensure they match the shown output.

## Enabling LLM Tests

If you want to enable the LLM-based tests, ensure you have a GPU and set
`TEST_LLM=1`, for example:
