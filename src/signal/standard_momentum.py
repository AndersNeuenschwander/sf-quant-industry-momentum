import polars as pl
import numpy as np
import datetime as dt
import sf_quant.data as sfd
import sf_quant.research as sfr
import polars_ols

start = dt.date(2012, 1, 1)
end = dt.date(2024, 12, 31)

columns = ['barrid', 'date', 'predicted_beta', 'return', 'specific_risk', 'price']

data = sfd.load_assets(
    start=start,
    end=end,
    in_universe=True,
    columns=columns
)

# calculate momentum

df = (data
    .with_columns(
        pl.col('return').truediv(100)
    )
    .sort('barrid', 'date')
    .with_columns(
        # compute rolling volatility using ewm_std with 22-day span
        pl.col('return')
        .ewm_std(span=22, min_samples=22)
        .over('barrid')
        .alias('vol')
    )
    .with_columns(
        # vol-scale the returns by dividing by variance
        (pl.col('return') / (pl.col('vol')))
        .alias('scaled_return')
    )
    .with_columns(
        pl.col('scaled_return')
        .rolling_sum(window_size=230)
        .over('barrid')
        .alias('momentum')
    )
    .with_columns(
        pl.col('momentum').shift(22).over('barrid')
    )
    # filtering
    .sort('barrid', 'date')
        .with_columns(
            pl.col('price').shift(1).over('barrid').alias('price_lag')
        )
        .filter(
            pl.col('price_lag').gt(5),
            pl.col('momentum').is_not_null()
        )
        .sort('barrid', 'date')
    # z-score momentum
    .with_columns(
            ((pl.col("momentum") - pl.col("momentum").mean().over("date")) /
            pl.col("momentum").std().over("date"))
            .alias("momentum_zscore")
        )
    # calculate alpha
    .with_columns(
            pl.col('momentum_zscore')
            .mul(0.5)
            .mul('specific_risk')
            .alias('alpha')
        )
    .with_columns(
    pl.col('momentum_zscore')
    .alias('signal')
    )
)

print(df)