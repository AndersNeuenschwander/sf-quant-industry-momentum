import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd


def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.

    Returns:
        pl.DataFrame: Market data with required columns
    """
    industry_cols = [
        "date",
        "barrid",
        "USSLOWL_AERODEF",
        "USSLOWL_AIRLINES",
        "USSLOWL_ALUMSTEL",
        "USSLOWL_APPAREL",
        "USSLOWL_AUTO",
        "USSLOWL_BANKS",
        "USSLOWL_BEVTOB",
        "USSLOWL_BIOLIFE",
        "USSLOWL_BLDGPROD",
        "USSLOWL_CHEM",
        "USSLOWL_CNSTENG",
        "USSLOWL_CNSTMACH",
        "USSLOWL_CNSTMATL",
        "USSLOWL_COMMEQP",
        "USSLOWL_COMPELEC",
        "USSLOWL_COMSVCS",
        "USSLOWL_CONGLOM",
        "USSLOWL_CONTAINR",
        "USSLOWL_DISTRIB",
        "USSLOWL_DIVFIN",
        "USSLOWL_ELECEQP",
        "USSLOWL_ELECUTIL",
        "USSLOWL_FOODPROD",
        "USSLOWL_FOODRET",
        "USSLOWL_GASUTIL",
        "USSLOWL_HLTHEQP",
        "USSLOWL_HLTHSVCS",
        "USSLOWL_HOMEBLDG",
        "USSLOWL_HOUSEDUR",
        "USSLOWL_INDMACH",
        "USSLOWL_INSURNCE",
        "USSLOWL_INTERNET",
        "USSLOWL_LEISPROD",
        "USSLOWL_LEISSVCS",
        "USSLOWL_LIFEINS",
        "USSLOWL_MEDIA",
        "USSLOWL_MGDHLTH",
        "USSLOWL_MULTUTIL",
        "USSLOWL_OILGSCON",
        "USSLOWL_OILGSDRL",
        "USSLOWL_OILGSEQP",
        "USSLOWL_OILGSEXP",
        "USSLOWL_PAPER",
        "USSLOWL_PHARMA",
        "USSLOWL_PRECMTLS",
        "USSLOWL_PSNLPROD",
        "USSLOWL_REALEST",
        "USSLOWL_RESTAUR",
        "USSLOWL_ROADRAIL",
        "USSLOWL_SEMICOND",
        "USSLOWL_SEMIEQP",
        "USSLOWL_SOFTWARE",
        "USSLOWL_SPLTYRET",
        "USSLOWL_SPTYCHEM",
        "USSLOWL_SPTYSTOR",
        "USSLOWL_TELECOM",
        "USSLOWL_TRADECO",
        "USSLOWL_TRANSPRT",
        "USSLOWL_WIRELESS",
    ]

    start = dt.date(2000, 1, 1)
    end = dt.date(2010, 12, 31)

    asset_columns = [
        'barrid', 
        'date', 
        'predicted_beta', 
        'return', 
        'specific_risk',
        'price'
    ]


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
        pl.when(pl.col(c).is_null()).then(pl.lit(0)).otherwise(pl.lit(1)).cast(pl.Int8).alias(c)
        for c in industry_only
    )

    # create the assets data frame

    assets_df = sfd.load_assets(
        start=start,
        end=end,
        columns=asset_columns,
        in_universe=True,
    )

    # Perform left merge on assets data frame with indsutry exposure data frame

    df = assets_df.join(
        industry_exposures_df, on=['date', 'barrid'], how='left'
        )

    # calculate log returns and shift them

    df = (df
        .with_columns(
            pl.col('return')
            .truediv(100)
        )
        .with_columns(
            pl.col('specific_risk')
            .truediv(100)
        )
        .with_columns(
            pl.col('return')
            .log1p()
            .alias('logreturn')
        )
        .sort(['date', 'barrid'])
    )

    # group by industry
    
    long = (
        df
        .unpivot(
            on=industry_only,          # columns to melt
            index=["barrid", "date", "return", "predicted_beta", "specific_risk", "price"],  # columns to keep as-is
            variable_name="industry",
            value_name="val",
        )
        .filter(pl.col("val") == 1)    # keep only rows where stock IS in that industry
        .drop("val")
        .sort(['date', 'industry'])
    )

    # Create Equal Weight Portfolio

    ew_port = (
        long
        .group_by(["date", "industry"])
        .agg(
            pl.col('return').mean().alias("ew_return"),
            )
        .sort(["industry", "date"])
    )

    return ew_port, long


def create_signal(df1, df2):
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # TODO: Load Data
    df1, df2 = load_data()

    ew_port = df1
    long = df2
    
    # Calculate Momentum
    

    industry_momentum = (
        ew_port
        .sort(['date', 'industry'])
        .with_columns(
            pl.col('ew_return')
            .rolling_sum(window_size=230)
            .over('industry')
            .alias('momentum')
        )
        .with_columns(
            pl.col('momentum')
            .shift(22)
            .over('industry')
        )
    )
    # go back to stock space and filter price

    long = (long
            .join(industry_momentum, on=['date', 'industry'], how='left')
            .filter(pl.col('price') >= 5)
            )
    
    # z-score momentum

    industry_momentum = long.with_columns(
        ((pl.col("momentum") - pl.col("momentum").mean().over("date")) / 
        pl.col("momentum").std().over("date"))
        .alias("score")
    )
    # compute alpha

    industry_momentum = (
        industry_momentum
        .with_columns(
            pl.col('score')
            .mul(.05)
            .mul(pl.col('specific_risk'))
            .alias('alpha')
        )
    )

    return industry_momentum

    # TODO: Add your signal logic here (remember alpha logic)

    # TODO: Save to data/signal.parquet

    pl.write_parquet(signal, output_path)

if __name__ == "__main__":
    create_signal()
