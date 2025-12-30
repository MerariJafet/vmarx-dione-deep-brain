.PHONY: setup help phasea report clean 

help:
	@echo "VMarx Dione DB - Project Commands"
	@echo "  setup    : Install dependencies"
	@echo "  phasea   : Run full Phase A evaluation suite"
	@echo "  report   : View final evaluation report summary"
	@echo "  clean    : Remove temp files and cache"

setup:
	pip install -r requirements.txt

phasea:
	python3 scripts/run_phase_a.py

report:
	cat reports/phaseA_portfolio_summary.md

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf logs/*.log
