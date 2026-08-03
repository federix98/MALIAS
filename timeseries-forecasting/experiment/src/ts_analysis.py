import numpy as np
import pandas as pd

from pathlib import Path

import logging
import traceback

import matplotlib.pyplot as plt
# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import split

from cfg import TEST_SIZE, VAL_SIZE, TS_PATH

from ts_utils import find_seasonality, find_trend_type, check_stationarity

from statsmodels.tsa.seasonal import seasonal_decompose

import argparse

def proc_data(ts, describe = False):

    ts = np.array(ts)

    train_ts, test_ts = split(ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")

    return {
        "ts": [train_ts, val_ts, test_ts],
    }

if __name__ == "__main__":

    # --- args ---
    parser = argparse.ArgumentParser(description="Time series programmatic analysis")
    args = parser.parse_args()


    df_path = TS_PATH / "ts.csv"
    logging.info(f"Dataset loading from {df_path}")
    df = pd.read_csv(df_path)
    logging.info(f"Dataset loaded.")

    res = []

    figs_path = Path("..") / "data" / "figs"
    figs_path.mkdir(exist_ok=True, parents=True)

    for (metric), sub_df in df.groupby(["metric"]):

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values

        processed = proc_data(ts, describe=True)

        ts_train, ts_eval, ts_test = processed["ts"][0], processed["ts"][1], processed["ts"][2]

        logging.info(f"Analysis of {metric=}")

        missing_values = sub_df.sort_values(by=["dt#"]).y.isna().sum()
        logging.info(f"Missing values: {missing_values}")
        missing_value_indexes = sub_df.sort_values(by=["dt#"]).loc[sub_df["y"].isna()]["dt#"].values
        logging.info(f"missing_value_indexes: {missing_value_indexes}")

        trend_type = None
        stationarity = None
        seasonality = None
        decomposed = False

        ts_analysis = np.concatenate((ts_train, ts_eval))
        logging.info(f"{ts_analysis.shape=}")


        try:
            seasonality = find_seasonality(ts_analysis)
            logging.info(f"Seasonality with statistical analysis: {seasonality}")
            if not missing_values: 
                if np.any(ts_analysis <= 0):
                    logging.info("Zero and negative values found in train and validation time series. Setting trend type to 'add'")
                    trend_type = 'add'
                else:
                    trend_type = find_trend_type(ts_analysis, seasonality)
                
                result = seasonal_decompose(ts_analysis, model=trend_type, period=seasonality)  # 'period' is the seasonal frequency

                # Plot the decomposition
                result.plot()
                plt.savefig(figs_path / f"seasonal_decomposition___app#metric_{metric}.png")
                decomposed = True

                # Access components
                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid

                stationarity = check_stationarity(ts_analysis)
        except Exception as e:
            logging.warning(f"Error during statistical analysis {e}")
            logging.warning("Traceback details:\n%s", traceback.format_exc())
            exit(1)
            

        res.append({
            "metric": metric,
            "total": len(ts),
            "train": len(ts_train),
            "eval": len(ts_eval),
            "test": len(ts_test),
            "missing_values": missing_values,
            "seasonality": seasonality,
            "stationarity": stationarity,
            "trend_type": trend_type,
            "decomposed": decomposed
        })

    save_path = Path("..") / "data" / "analysis"
    save_path.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(res).to_csv(save_path / f"ts_analysis_{args.aggregation}.csv", index=None)