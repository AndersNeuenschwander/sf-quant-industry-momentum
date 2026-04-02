import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd


def create_signal():
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal/industry_momentum.parquet")
    # output_path = os.getenv("SIGNAL_PATH", "data/signal/standard_momentum.parquet")
    # output_path = os.getenv("SIGNAL_PATH", "data/signal/idiosyncratic_momentum.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configurations

    industry_cols = [
            "date", "barrid",
            "USSLOWL_AERODEF", "USSLOWL_AIRLINES", "USSLOWL_ALUMSTEL", "USSLOWL_APPAREL",
            "USSLOWL_AUTO", "USSLOWL_BANKS", "USSLOWL_BEVTOB", "USSLOWL_BIOLIFE",
            "USSLOWL_BLDGPROD", "USSLOWL_CHEM", "USSLOWL_CNSTENG", "USSLOWL_CNSTMACH",
            "USSLOWL_CNSTMATL", "USSLOWL_COMMEQP", "USSLOWL_COMPELEC", "USSLOWL_COMSVCS",
            "USSLOWL_CONGLOM", "USSLOWL_CONTAINR", "USSLOWL_DISTRIB", "USSLOWL_DIVFIN",
            "USSLOWL_ELECEQP", "USSLOWL_ELECUTIL", "USSLOWL_FOODPROD", "USSLOWL_FOODRET",
            "USSLOWL_GASUTIL", "USSLOWL_HLTHEQP", "USSLOWL_HLTHSVCS", "USSLOWL_HOMEBLDG",
            "USSLOWL_HOUSEDUR", "USSLOWL_INDMACH", "USSLOWL_INSURNCE", "USSLOWL_INTERNET",
            "USSLOWL_LEISPROD", "USSLOWL_LEISSVCS", "USSLOWL_LIFEINS", "USSLOWL_MEDIA",
            "USSLOWL_MGDHLTH", "USSLOWL_MULTUTIL", "USSLOWL_OILGSCON", "USSLOWL_OILGSDRL",
            "USSLOWL_OILGSEQP", "USSLOWL_OILGSEXP", "USSLOWL_PAPER", "USSLOWL_PHARMA",
            "USSLOWL_PRECMTLS", "USSLOWL_PSNLPROD", "USSLOWL_REALEST", "USSLOWL_RESTAUR",
            "USSLOWL_ROADRAIL", "USSLOWL_SEMICOND", "USSLOWL_SEMIEQP", "USSLOWL_SOFTWARE",
            "USSLOWL_SPLTYRET", "USSLOWL_SPTYCHEM", "USSLOWL_SPTYSTOR", "USSLOWL_TELECOM",
            "USSLOWL_TRADECO", "USSLOWL_TRANSPRT", "USSLOWL_WIRELESS",
        ]

    industry_only = [c for c in industry_cols if c not in ["date", "barrid"]]

    start = dt.date(2012, 1, 1)
    end = dt.date(2024, 12, 31)

    asset_columns = ['barrid', 'date', 'predicted_beta', 'return', 'specific_risk', 'price']

    # create the assets data frame

    assets_df = sfd.load_assets(
        start=start,
        end=end,
        columns=asset_columns,
        in_universe=True,
    )

    assets_df

    # Create Industry_exposure_df
    # 0 indicates that stock is NOT in the industry and 1 indicates the stock IS in the industry
    industry_exposures_df = sfd.load_exposures(
        start=start, 
        end=end, 
        in_universe=True,
        columns=industry_cols
    )

    industry_only = [c for c in industry_cols if c not in ["date", "barrid"]]

    industry_exposures_df = industry_exposures_df.with_columns(
        pl.col(c).fill_null(0).alias(c)
        for c in industry_only
    )

    # normalize factor exposures

    row_sum_expr = pl.sum_horizontal(industry_only)

    industry_exposures_df = industry_exposures_df.with_columns(
        (pl.col(c) / row_sum_expr).alias(c)
        for c in industry_only
    )

    # filter out the 0 values before unpivot to make join lighter down the script

    industry_exposures_df = (
        industry_exposures_df
        .filter(row_sum_expr > 0)          # <-- drop all-zero rows before the explosion
        .unpivot(
            on=industry_only,
            index=['date', 'barrid'],
            variable_name='industry',
            value_name='exposures'
        )
        .filter(pl.col('exposures') > 0)
        .fill_nan(0)   # <-- drop zero-exposure industry columns per row
    )

    # create factor returns data frame

    factor_returns_df = sfd.load_factors(
        start=start, 
        end=end, 
    )

    # do some chill unpivoting

    factor_returns_df = (
        factor_returns_df                          # wide: barrid, date, 59 industry cols
        .unpivot(
            on=industry_only,
            index=["date"],                 # only carry the keys through the explosion
            variable_name="industry",
            value_name="return",
        )
    )

    # calculate momentum
    industry_momentum = (
        factor_returns_df
        .sort(['date', 'industry']) # date, barrid for standard momentum and idiosyncratic momentum (specific_return)
        .with_columns(
            # compute rolling volatility using ewm_std with 22-day span
            pl.col('return')
            .ewm_std(span=22)
            .over('industry')
            .alias('vol')
        )
        .with_columns(
            # vol-scale the returns by dividing by variance
            (pl.col('return') / (pl.col('vol')))
            .alias('scaled_return')
        )
        .with_columns(
            # sum the vol-scaled returns over the lookback window
            pl.col('scaled_return')
            .rolling_sum(window_size=230)
            .over('industry')
            .alias('momentum')
        )
        .with_columns(
            pl.col('momentum')
            .shift(22)
            .over('industry')
        )
        .drop_nulls()
    )

    # join the two data frames

    df = (
        industry_exposures_df
        .join(industry_momentum, on=["date", "industry"], how="left")
        .group_by(["date", "barrid"])
        .agg(
            (pl.col("exposures") * pl.col("momentum")).sum().alias("momentum_score")
        )


        # z-score cross-sectionally within each date

        .with_columns(
            ((pl.col("momentum_score") - pl.col("momentum_score").mean().over("date")) /
            pl.col("momentum_score").std().over("date"))
            .alias("momentum_zscore")
        )
    )

    df = (
        df
        .join(
            assets_df,
            on=['barrid', 'date'],
            how='left'
        )
        .with_columns(
            pl.col('return')
            .truediv(100)
        )
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
        .with_columns(
            pl.col('price')
            .shift(1)
            .over('barrid')
            .alias('price_lag')
        )
        .filter(
            pl.col('price_lag') >=5,
            pl.col('alpha').is_not_nan()
        )
    )


    ### standard_momentum signal code

    # columns = ['barrid', 'date', 'predicted_beta', 'return', 'specific_risk', 'price']

    # data = sfd.load_assets(
    #     start=start,
    #     end=end,
    #     in_universe=True,
    #     columns=columns
    # )

    # # calculate momentum

    # df = (data
    #     .with_columns(
    #         pl.col('return').truediv(100)
    #     )
    #     .sort('barrid', 'date')
    #     .with_columns(
    #         # compute rolling volatility using ewm_std with 22-day span
    #         pl.col('return')
    #         .ewm_std(span=22, min_samples=22)
    #         .over('barrid')
    #         .alias('vol')
    #     )
    #     .with_columns(
    #         # vol-scale the returns by dividing by variance
    #         (pl.col('return') / (pl.col('vol')))
    #         .alias('scaled_return')
    #     )
    #     .with_columns(
    #         pl.col('scaled_return')
    #         .rolling_sum(window_size=230)
    #         .over('barrid')
    #         .alias('momentum')
    #     )
    #     .with_columns(
    #         pl.col('momentum').shift(22).over('barrid')
    #     )
    #     # filtering
    #     .sort('barrid', 'date')
    #         .with_columns(
    #             pl.col('price').shift(1).over('barrid').alias('price_lag')
    #         )
    #         .filter(
    #             pl.col('price_lag').gt(5),
    #             pl.col('momentum').is_not_null()
    #         )
    #         .sort('barrid', 'date')
    #     # z-score momentum
    #     .with_columns(
    #             ((pl.col("momentum") - pl.col("momentum").mean().over("date")) /
    #             pl.col("momentum").std().over("date"))
    #             .alias("momentum_zscore")
    #         )
    #     # calculate alpha
    #     .with_columns(
    #             pl.col('momentum_zscore')
    #             .mul(0.5)
    #             .mul('specific_risk')
    #             .alias('alpha')
    #         )
    #     .with_columns(
    #     pl.col('momentum_zscore')
    #     .alias('signal')
    #     )
    # )

    ### idiosyncratic_momentum signal code

    # columns = ['barrid', 'date', 'predicted_beta', 'return', 'specific_risk', 'price', 'specific_return']

    # data = sfd.load_assets(
    #     start=start,
    #     end=end,
    #     in_universe=True,
    #     columns=columns
    # )

    # # calculate momentum

    # df = (data
    #     .with_columns(
    #         pl.col('specific_return').truediv(100)
    #     )
    #     .sort('barrid', 'date')
    #     .with_columns(
    #         # compute rolling volatility using ewm_std with 22-day span
    #         pl.col('specific_return')
    #         .ewm_std(span=22, min_samples=22)
    #         .over('barrid')
    #         .alias('vol')
    #     )
    #     .with_columns(
    #         # vol-scale the returns by dividing by variance
    #         (pl.col('specific_return') / (pl.col('vol')))
    #         .alias('scaled_return')
    #     )
    #     .with_columns(
    #         pl.col('scaled_return')
    #         .rolling_sum(window_size=230)
    #         .over('barrid')
    #         .alias('momentum')
    #     )
    #     .with_columns(
    #         pl.col('momentum').shift(22).over('barrid')
    #     )
    #     # filtering
    #     .sort('barrid', 'date')
    #         .with_columns(
    #             pl.col('price').shift(1).over('barrid').alias('price_lag')
    #         )
    #         .filter(
    #             pl.col('price_lag').gt(5),
    #             pl.col('momentum').is_not_null()
    #         )
    #         .sort('barrid', 'date')
    #     # z-score momentum
    #     .with_columns(
    #             ((pl.col("momentum") - pl.col("momentum").mean().over("date")) /
    #             pl.col("momentum").std().over("date"))
    #             .alias("momentum_zscore")
    #         )
    #     # calculate alpha
    #     .with_columns(
    #             pl.col('momentum_zscore')
    #             .mul(0.5)
    #             .mul('specific_risk')
    #             .alias('alpha')
    #         )
    #     .with_columns(
    #     pl.col('momentum_zscore')
    #     .alias('signal')
    #     )
    # )

    df.write_parquet(output_path)



if __name__ == "__main__":
    create_signal()
