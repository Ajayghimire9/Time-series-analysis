PYTHON ?= python

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

run:
	$(PYTHON) -m src.pipeline
