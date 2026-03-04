# SF-Research

A template project for developing, researching, and backtesting trading signals.

## 🚀 How to Use This Template
#### To create your own repository using this template, follow these quick steps:
1. Click the Button: At the top of this repository page, click the green "Use this template" button and select "Create a new repository."
2. Configure:
   - Choose an Owner (your account).
   - Give your new repository a Name, "sf-research-{signal_name}"
3. Create: Click "Create repository from template."
4. Clone: Once your new repo is ready, clone it to your local machine:
```bash
git clone https://github.com/your-username/your-new-repo.git
```

## Project Structure

```
sf-signal/
├── src/
│   ├── framework/
│   │   ├── ew_dash.py            # Equal-weight dashboard (do not edit)
│   │   ├── opt_dash.py           # Optimal portfolio dashboard (do not edit)
│   │   └── backtest.py       # Run the backtest (edit config only)
│   └── signal/
│       └── signal.py      # Your signal implementation (edit this)
├── data/
│   ├── signal.parquet            # Output: Your signal
│   └── weights/                  # Output: Backtest weights
└── README.md
```

## Workflow

### 1. **Implement Signal** (`signal.py`)
   - Customize date ranges, data columns, and calculation logic
   - Develop your signal logic
   - Saves signal to `data/signal.parquet`

   ```bash
   make create-signal
   ```

### 2. **View Equal-Weight Performance** (`ew_dash.py`)
   - Compare your signal against an equal-weight baseline
   - Analyze signal characteristics
   - Visualize signal properties and performance

   ```bash
   make ew-dash
   ```

### 3. **Run Backtest** (`backtest.py`)
   - Run MVO-based backtest on your signal
   - Generates optimal portfolio weights
   - Saves results to `data/weights.parquet`

   ```bash
   make backtest
   ```

### 4. **View Optimized Performance** (`opt_dash.py`)
   - View optimized portfolio performance
   - Analyze backtest returns, drawdowns, and metrics

   ```bash
   make opt-dash
   ```

## Data Files

All data files are stored in the `data/` directory:

- **`data/signal.parquet`**: Output from `signal.py`
  - Columns: `date`, `barrid`, `alpha` (your signal)
  - Format: Parquet (AlphaSchema)

- **`data/weights/*.parquet`**: Output from backtest
  - Contains: Portfolio weights and performance data
  - Format: Parquet

## Quick Start

```bash
# 1. Implement your signal
# Edit src/signal/signal.py with your logic
make create-signal

# 2. View equal-weight performance
make ew-dash

# 3. Run backtest
make backtest

# 4. View optimized performance
make opt-dash
```

## Template Files (Do Not Need to Edit)

The following files are templates and should not be modified:
- `src/framework/ew_dash.py` - Equal-weight comparison dashboard
- `src/framework/opt_dash.py` - Optimized portfolio dashboard
- `src/framework/backtest.py` - Backtest runner

If you want to edit the marimo notebooks use:
```bash
uv run marimo edit src/framework/{}_dash.py
```

**All signal customization happens in `src/signal/signal.py`.**

## Configuration

Update `src/framework/backtest.py` if needed:
- `byu_email`: Your BYU email for job submission
- `gamma`: Transaction costs or risk aversion parameter
- `constraints`: Add portfolio constraints
- `slurm`: Adjust computational resources

## Next Steps

1. Implement your signal logic in `src/signal/signal.py`
2. Run `make create-signal` to generate your signal
3. Compare against baseline with `make ew-dash`
4. Run backtest with `make backtest`
5. Analyze optimized results with `make opt-dash`
6. Iterate and refine your approach

---

**Note**: This is a template project. Customize `src/signal/signal.py` with your unique signal logic, then use the workflow above to backtest your ideas.


