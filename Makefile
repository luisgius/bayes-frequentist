PYTHON := .venv/bin/python
UV := uv
TORCH_INDEX := https://download.pytorch.org/whl/cpu

.PHONY: setup test test-all lint

setup:
	$(UV) venv --python 3.12 --allow-existing .venv
	$(UV) pip install --python $(PYTHON) --index-url $(TORCH_INDEX) torch==2.14.1
	$(UV) pip install --python $(PYTHON) -r requirements.txt
	$(UV) pip install --python $(PYTHON) -e .

test:
	$(PYTHON) -m pytest -q

test-all:
	$(PYTHON) -m pytest -o addopts= -q

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
