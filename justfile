default:
  @just --list

# Install dependencies
requirements:
  python -m pip install -U uv pip setuptools wheel
  python -m uv pip install -r .devcontainer/requirements.txt
  pre-commit install

# Delete all compiled files and cache
clean:
  find . -type f -name "*.py[co]" -delete
  find . -type f -name "__pycache__" -delete
  rm -rf .mypy_cache/
  rm -rf .pytest_cache/
  rm -rf .ruff_cache/

# Testing
test:
  pytest --durations=0 -vv .

# Basic linting
lint:
  black src
  ruff check src
  mypy src
