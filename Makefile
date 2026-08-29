.PHONY: install install-dev test lint format classical neural all clean

PY ?= python
SRC := src/prosody

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,notebooks]"
	pre-commit install

test:
	pytest -q

lint:
	ruff check $(SRC) tests scripts

format:
	black $(SRC) tests scripts
	ruff check --fix $(SRC) tests scripts

classical:
	$(PY) scripts/run_classical_only.py --data-root AutoRPT_Data \
	    --cache artifacts/corpus.pkl \
	    --val-speaker m2b --test-speaker f3a \
	    --out artifacts/classical_results.json -v

neural:
	$(PY) scripts/run_neural.py --data-root AutoRPT_Data \
	    --cache artifacts/corpus.pkl \
	    --val-speaker m2b --test-speaker f3a \
	    --epochs 12 --batch-size 64 \
	    --out artifacts/neural_results.json -v

all: classical neural

clean:
	rm -rf artifacts/*.pkl artifacts/*.pt artifacts/*.json
