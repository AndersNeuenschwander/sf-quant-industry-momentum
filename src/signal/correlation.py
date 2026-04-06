import datetime as dt
from pathlib import Path
import great_tables as gt

import polars as pl
import sf_quant.data as sfd

start = dt.date(2012, 1, 1)
end = dt.date(2024, 12, 31)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

results_folder = BASE_DIR / "results"
results_folder.mkdir(exist_ok=True)

# load weight files 
standard_momentum_weights = pl.read_parquet(str(BASE_DIR / "data/weights/standard_momentum/*.parquet"))
industry_momentum_weights = pl.read_parquet(str(BASE_DIR / "data/weights/industry_momentum/*.parquet"))
idiosyncratic_momentum_weights = pl.read_parquet(str(BASE_DIR / "data/weights/idiosyncratic_momentum/*.parquet"))

# Get returns

returns = (
    sfd.load_assets(
        start=start, end=end, columns=["date", "barrid", "return"], in_universe=True
    )
    .sort("date", "barrid")
    .select(
        "date",
        "barrid",
        (pl.col("return").truediv(100)).alias("return"),
    )
)


# compute portfolio returns

standard_mom_port_returns = (
    standard_momentum_weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("return").mul(pl.col("weight")).sum().alias("standard_mom_return"))
    .sort("date")
)

industry_mom_port_returns =  (
    industry_momentum_weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("return").mul(pl.col("weight")).sum().alias("industry_mom_return"))
    .sort("date")
)

idiosyncratic_mom_port_returns = (
    idiosyncratic_momentum_weights.join(other=returns, on=['date', 'barrid'], how='left')
    .group_by('date')
    .agg(pl.col("return").mul(pl.col("weight")).sum().alias("idiosyncratic_mom_return"))
    .sort('date')
)

# compute portfolio returns

portfolio_returns = (
    standard_mom_port_returns
    .join(other=industry_mom_port_returns, on=["date"], how="left")
    .join(other=idiosyncratic_mom_port_returns, on=["date"], how="left")
    .sort("date")
    .select(["date", "standard_mom_return", "industry_mom_return", "idiosyncratic_mom_return"])
)


# Create summary table with all pairwise correlations
summary = portfolio_returns.select(
    pl.corr("standard_mom_return", "industry_mom_return").alias("standard_vs_industry"),
    pl.corr("standard_mom_return", "idiosyncratic_mom_return").alias("standard_vs_idiosyncratic"),
    pl.corr("industry_mom_return", "idiosyncratic_mom_return").alias("industry_vs_idiosyncratic"),
)

table = (
    gt.GT(summary)
    .tab_header(title="Momentum Strategy Correlation Table")
    .cols_label(
        standard_vs_industry="Standard vs Industry",
        standard_vs_idiosyncratic="Standard vs Idiosyncratic",
        industry_vs_idiosyncratic="Industry vs Idiosyncratic",
    )
    .fmt_number(
        ["standard_vs_industry", "standard_vs_idiosyncratic", "industry_vs_idiosyncratic"],
        decimals=2
    )
    .opt_stylize(style=4, color="gray")
)


table_path = results_folder / "correlation_table.html"
with open(table_path, "w") as f:
    f.write(table.as_raw_html())

print(f"Table saved to {table_path}")
print(summary)