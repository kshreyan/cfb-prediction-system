.PHONY: setup test leakage-test backtest market-eval ensemble-eval gbm-eval spread-eval total-eval predict report deploy lint

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
	$(VENV)/python -m cfb.cli backtest --start-season 2015 --end-season 2025

market-eval:
	$(VENV)/python -m cfb.cli market-eval --start-season 2015 --end-season 2025

ensemble-eval:
	$(VENV)/python -m cfb.cli ensemble-eval --start-season 2015 --end-season 2025

gbm-eval:
	$(VENV)/python -m cfb.cli gbm-eval --start-season 2015 --end-season 2025

spread-eval:
	$(VENV)/python -m cfb.cli spread-eval --start-season 2015 --end-season 2025

total-eval:
	$(VENV)/python -m cfb.cli total-eval --start-season 2015 --end-season 2025

predict:
	$(VENV)/python -m cfb.cli predict

report:
	$(VENV)/python -m cfb.cli report --out docs

deploy: report
	@echo "git add docs && git commit -m 'Rebuild site' && git push  (GitHub Pages serves docs/ on main)"
