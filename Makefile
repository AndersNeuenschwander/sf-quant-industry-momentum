.PHONY: ew-dash opt-dash create-signal backtest

ew-dash:
	uv run marimo run src/framework/ew_dash.py

opt-dash:
	uv run marimo run src/framework/opt_dash.py

create-signal:
	uv run python src/signal/signal.py

backtest:
	uv run python src/framework/backtest.py
