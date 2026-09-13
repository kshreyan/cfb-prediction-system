.PHONY: setup test leakage-test backtest predict report deploy lint

VENV := .venv/bin

setup:
	python3 -m venv .venv
	$(VENV)/pip install -q --upgrade pip
	$(VENV)/pip install -q -e ".[dev]"

test:
	$(VENV)/python -m pytest tests/ -v

leakage-test:
	$(VENV)/python -m pytest tests/leakage -v

lint:
	$(VENV)/python -m ruff check src tests
	$(VENV)/python -m mypy src

check-cfbd:
	$(VENV)/python -m cfb.cli check-cfbd

backtest:
	$(VENV)/python -m cfb.cli backtest --seasons 2015-2024

predict:
	$(VENV)/python -m cfb.cli predict --week current

report:
	$(VENV)/python -m cfb.cli report --out docs/

deploy: report
	@echo "Push docs/ to gh-pages (see docs/DEPLOY.md)"
