# Industry Momentum: Perpendicular Alpha
**Research Report** | Anders Neuenschwander and Brandon Waits | April 10, 2026

---

## 1. Summary

This report presents the research and results of an industry momentum signal strategy. The core idea is to capture momentum at the industry factor level rather than at the individual stock level, then project those factor-level momentum scores back down to individual assets via industry exposure weights. This creates a signal that is largely orthogonal — or "perpendicular" — to standard stock-level and idiosyncratic momentum strategies.

The signal is constructed by computing volatility-scaled rolling momentum for each of 59 US equity industries (using a 230-day lookback and 22-day lag), then weighting each stock's score by its exposure to those industries. The resulting per-stock signal is z-scored cross-sectionally and used as an alpha in a Mean-Variance Optimization (MVO) backtest.

### Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Sharpe Ratio (Industry Momentum) | 0.83 | Full sample 2013–2024 |
| Total Return | 65.25% | Annualized across 3,018 trading days |
| Mean Annual Return | 4.33% | |
| Volatility | 5.25% | Annualized |
| Max Drawdown | -10.07% | |
| IC (mean) | 0.0163 | Rank IC vs. next-day returns |
| IC IR | 0.151 | IC mean / IC std |
| Correlation vs. Standard Momentum | 0.25 | Signal-level correlation |
| Correlation vs. Idiosyncratic Momentum | 0.13 | Signal-level correlation |

---

## 2. Data Requirements

### Sources
- Asset-level data: daily returns, specific risk, predicted beta, price, specific return — loaded via `sf_quant.data.load_assets()`
- Industry exposure data: binary (0/1) flags for each of 59 USSLOWL industry buckets per stock per date — loaded via `sf_quant.data.load_exposures()`
- Factor return data: daily returns for each industry factor — loaded via `sf_quant.data.load_factors()`

### Date Range
- Start: January 1, 2012
- End: December 31, 2024
- Effective signal start (after lookback burn-in): ~February 2013

### Inputs Required
- `barrid`: unique security identifier
- `date`: trading date
- `return`: total daily stock return (%)
- `specific_return`: idiosyncratic daily return (used in idiosyncratic momentum variant)
- `specific_risk`: Barra-style specific risk estimate
- `predicted_beta`: predicted market beta
- `price`: closing price (used for filtering stocks below $5)
- `USSLOWL_*` columns: 59 industry exposure flags

### Preprocessing
- Industry exposures are filled with 0 for nulls and then normalized so each stock's industry weights sum to 1 (handling partial industry memberships)
- Factor returns are unpivoted from wide format (one column per industry) to long format (one row per industry per date)
- Stocks with a lagged price below $5 are excluded to remove low-priced/illiquid names
- Alpha values that are NaN are filtered out
- The final alpha is scaled by `0.5 * specific_risk` to convert the z-scored signal into units compatible with the MVO optimizer

---

## 3. Approach / System Design

### Economic Intuition

Standard price momentum is one of the most well-documented anomalies in finance: stocks that performed well over the prior 6–12 months tend to continue outperforming in the short run. However, a large portion of this effect is driven by industry-level trends. When a macro event (a commodity shock, a regulatory change, a technological wave) hits an industry, prices across all stocks in that sector drift in the same direction for months as investors slowly update their beliefs.

This means standard momentum strategies, while appearing diversified across individual stocks, are really implicitly betting on a handful of industry trends. The industry momentum strategy makes this bet explicitly and cleanly: instead of ranking individual stocks, we rank industries by their recent volatility-adjusted performance and then assign scores to stocks based on their membership in those industries.

The key insight — and the source of the "Perpendicular Alpha" name — is that the industry momentum signal is largely decorrelated from both standard momentum (0.25 correlation) and idiosyncratic momentum (0.13 correlation). This means it adds genuine diversification when combined with those signals in a portfolio.

### Signal Construction — Step by Step

**Step 1: Compute industry-level momentum.** For each of the 59 industries, compute a volatility-scaled rolling sum of daily factor returns over a 230-day window, lagged by 22 days to avoid the short-term reversal effect. Dividing by the exponentially-weighted 22-day standard deviation of factor returns normalizes each industry's score for its own risk level — so a high-volatility sector like biotech and a low-volatility sector like utilities are on comparable scales.

**Step 2: Project down to the asset level.** Each stock's momentum score is the exposure-weighted sum of industry momentum scores across all industries it belongs to. This is a matrix multiplication:

```
s_t      =   X_t     ×   m_t
(N×1)       (N×K)       (K×1)
```

where `X_t` is the exposure matrix (stocks by industries) and `m_t` is the vector of industry momentum scores.

**Step 3: Z-score cross-sectionally.** On each date, standardize the momentum scores across all stocks in the universe to have mean 0 and standard deviation 1. This removes any overall market-wide level shift.

**Step 4: Convert to alpha.** Multiply the z-score by 0.5 and by the stock's specific risk. This converts the dimensionless z-score into an expected return (alpha) in units that the MVO optimizer understands, where the 0.5 scalar controls signal aggressiveness.

### Comparison with Variants

| Signal | What it measures | Sharpe | Total Return |
|--------|-----------------|--------|-------------|
| Standard Momentum | Individual stock price momentum | 0.52 | 34.44% |
| Idiosyncratic Momentum | Stock-specific (market-neutral) momentum | 0.50 | 33% |
| **Industry Momentum** | **Industry factor momentum projected to stocks** | **0.83** | **65.25%** |

Industry momentum substantially outperforms both alternatives on a risk-adjusted basis, while remaining largely uncorrelated with them — making it a strong candidate for signal combination.

---

## 4. Code Structure

The project follows the standard sf-signal framework structure:

```
sf-signal/
├── src/
│   ├── framework/
│   │   ├── ew_dash.py            # Equal-weight dashboard (do not edit)
│   │   ├── opt_dash.py           # Optimal portfolio dashboard (do not edit)
│   │   └── run_backtest.py       # Run the backtest (edit config only)
│   └── signal/
│       └── create_signal.py      # Signal implementation (edit this)
├── data/
│   ├── signal.parquet            # Output: signal file
│   └── weights/                  # Output: backtest weights
└── README.md
```

### Step 1 — Implement Signal (`create_signal.py`)

This is the primary file to edit. It loads assets, industry exposures, and factor returns; computes volatility-scaled industry momentum; projects scores to the asset level; z-scores them; and saves the final signal to `data/signal.parquet`.

```bash
make create-signal
```

### Step 2 — View Equal-Weight Performance (`ew_dash.py`)

Launches an interactive Marimo dashboard that reads `signal.parquet` and produces signal statistics, a distribution plot, quantile portfolio cumulative returns, IC metrics, and a Fama-French regression. Use this to evaluate the raw signal before running the optimizer.

```bash
make ew-dash
```

### Step 3 — Run Backtest (`run_backtest.py`)

Submits an MVO-based backtest job (via Slurm) using the signal. Configuration (gamma, constraints, email, paths) is controlled via environment variables in a `.env` file. Output is year-by-year weight parquet files saved to `data/weights/`.

```bash
make backtest
```

### Step 4 — View Optimized Performance (`opt_dash.py`)

Loads the output weights and evaluates the full optimized portfolio: cumulative returns, drawdown, leverage, turnover, IC, and Fama-French regression.

```bash
make opt-dash
```

---

## 5. Results / Evaluation

### 5.1 Industry Momentum — Optimized Backtest

The following results are from the MVO backtest of the industry momentum signal over the full 2013–2024 sample period (3,018 trading days).

**Performance Summary**

| Metric | Value |
|--------|-------|
| Count (trading days) | 3,018 |
| Mean Annual Return | 4.33% |
| Volatility | 5.25% |
| Total Return | 65.25% |
| Sharpe Ratio | 0.83 |

**Drawdown Summary**

| Metric | Value |
|--------|-------|
| Mean Drawdown | -2.89% |
| Max Drawdown | -10.07% |
| Current Drawdown | -3.57% |
| Longest Drawdown (days) | 793 |

**Leverage Summary**

| Metric | Value |
|--------|-------|
| Mean Leverage | 359.31x |
| Min Leverage | 255.55x |
| Max Leverage | 455.54x |
| Std Leverage | 53.16x |

**Cumulative Returns**

![Industry Momentum — Portfolio Cumulative Log Returns](chart_industry_cumulative_returns.png)

**Drawdown**

![Industry Momentum — Portfolio Drawdown](chart_industry_drawdown.png)

**Information Coefficient (IC)**

| Metric | Value |
|--------|-------|
| IC (mean) | 0.0163 |
| IC IR | 0.151 |

![Industry Momentum — Cumulative Information Coefficient](chart_industry_cumulative_ic.png)

---

### 5.2 Signal Correlation Analysis

One of the primary motivations for the industry momentum signal is its low correlation with existing signals, suggesting it captures a genuinely different source of return.

| Pair | Signal Correlation | Portfolio Return Correlation |
|------|-------------------|------------------------------|
| Industry vs. Standard Momentum | 0.25 | 0.25 |
| Industry vs. Idiosyncratic Momentum | 0.13 | 0.13 |
| Standard vs. Idiosyncratic Momentum | N/A | 0.79 |

The near-zero correlation with idiosyncratic momentum (0.13) is particularly notable: idiosyncratic momentum strips out common factor movements, while industry momentum explicitly exploits them. These two strategies are nearly orthogonal in portfolio space, making them natural complements.

---

## 6. Performance Discussion

### Strengths
- Substantially higher Sharpe ratio (0.83) compared to standard (0.52) and idiosyncratic (0.50) momentum, with a much higher total return (65% vs. ~34%)
- Low correlation to existing signals means strong diversification benefit when combined into a composite alpha
- The vol-scaling step (dividing by EWM standard deviation) makes the signal robust to differences in industry-level volatility regimes, preventing noisy periods from dominating the score
- The 22-day lag avoids the well-known short-term reversal effect that plagues shorter-horizon momentum strategies

### Weaknesses
- The max drawdown of -10.07% and longest drawdown of 793 days indicate meaningful tail risk and recovery periods — strategy can be underwater for multi-year stretches
- High leverage (mean ~360x) is characteristic of long-short equity strategies in a Barra framework but may present implementation challenges in real portfolios
- Relatively low IC (0.0163) means the signal has limited per-day predictive power; the alpha comes from consistent small edges over many trades

---

## 7. Limitations

- The backtest assumes the Barra risk model's industry classifications remain stable, but industries can evolve (e.g., internet companies migrating between tech and media buckets)
- Transaction costs and market impact are not modeled in the backtest; the high leverage and turnover imply significant implementation shortfall in practice
- The signal relies on availability of both the factor returns and the exposure matrix from the Barra risk model — it cannot be replicated without those proprietary data sources
- The 22-day skip and 230-day lookback are not optimized out-of-sample; there is some risk of look-ahead in parameter selection
- The correlation analysis is performed on the full backtest period; correlation may vary significantly across market regimes

---

## 8. Future Work

- **Ratio constraints:** Constrain the backtest so that the long and short legs within each industry maintain a fixed ratio, more closely replicating a pure factor momentum tilt
- **Signal combination:** Build a composite alpha by blending industry momentum with idiosyncratic momentum (low correlation suggests near-additive Sharpe improvements)
- **Regime conditioning:** Evaluate whether the signal's performance varies across economic regimes (e.g., expansion vs. recession, high vs. low dispersion environments)
- **Lookback optimization:** Test alternative lookback windows and lag periods in a hold-out sample to assess robustness
- **Cost-aware optimization:** Incorporate estimated transaction costs into the MVO objective to reduce turnover and improve net-of-cost performance

---

## Appendix

### A. Standard Momentum Results

| Metric | Value |
|--------|-------|
| Date Range | 2013-02-04 to 2024-12-31 (2,998 days) |
| Mean Annual Return | 2.61% |
| Volatility | 5% |
| Total Return | 34.44% |
| Sharpe Ratio | 0.52 |
| Max Drawdown | -12.6% |
| Longest Drawdown (days) | 575 |
| Mean Leverage | 401.2x |

![Standard Momentum — Portfolio Cumulative Log Returns](chart_standard_cumulative_returns.png)

![Standard Momentum — Portfolio Drawdown](chart_standard_drawdown.png)

---

### B. Idiosyncratic Momentum Results

| Metric | Value |
|--------|-------|
| Date Range | 2013-02-04 to 2024-12-31 (2,998 days) |
| Mean Annual Return | 2.52% |
| Volatility | 5% |
| Total Return | 33% |
| Sharpe Ratio | 0.50 |
| Max Drawdown | -8.55% |
| Longest Drawdown (days) | 576 |
| Mean Leverage | 449.43x |

![Idiosyncratic Momentum — Portfolio Cumulative Log Returns](chart_idiosyncratic_cumulative_returns.png)

![Idiosyncratic Momentum — Portfolio Drawdown](chart_idiosyncratic_drawdown.png)

---

### C. Reproducibility Notes

- Set `SIGNAL_PATH=data/signal/industry_momentum.parquet` in your `.env` file to run the industry momentum variant
- Run `make create-signal`, then `make backtest`, then `make opt-dash`
- For standard momentum, comment/uncomment the relevant blocks in `create_signal.py` and update `SIGNAL_PATH` accordingly
- Python dependencies: `polars`, `sf_quant`, `sf_backtester`, `polars_ols`, `python-dotenv`
