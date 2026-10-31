PYTHON := .venv/bin/python
UV := uv
TORCH_INDEX := https://download.pytorch.org/whl/cpu

.PHONY: setup test test-all lint path-a path-b notebook figures all

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

path-a:
	$(PYTHON) scripts/run_path_a.py

path-b:
	$(PYTHON) scripts/run_path_b.py

notebook:
	$(PYTHON) -m nbconvert --execute --to notebook --inplace notebooks/01_holy_war_reconciliation.ipynb

figures:
	$(PYTHON) scripts/run_path_a.py --figures-only
	$(PYTHON) scripts/run_path_b.py --figures-only
	$(PYTHON) scripts/make_summary_figure.py

all:
	$(MAKE) lint test
	$(MAKE) path-a
	$(MAKE) path-b
	$(PYTHON) scripts/make_summary_figure.py
	$(MAKE) notebook
