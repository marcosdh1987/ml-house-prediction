# load the dataset
import pandas as pd


def get_metrics():

    query = f"""
    SELECT
        key, value
    FROM
        latest_metrics
    """

    df = pd.read_sql(query, "sqlite:///src/mlflow/mlruns.db")

    metrics = {}
    metrics["test"] = (
        df[df["key"].isin(["test_rmae", "test_mae"])]
        .set_index("key")
        .to_dict()["value"]
    )
    # round the values 3 decimals
    metrics["test"]["test_rmae"] = round(metrics["test"]["test_rmae"], 3)
    metrics["test"]["test_mae"] = round(metrics["test"]["test_mae"], 3)
    metrics["train"] = (
        df[df["key"].isin(["train_rmae", "train_mae"])]
        .set_index("key")
        .to_dict()["value"]
    )
    # round the values 3 decimals
    metrics["train"]["train_rmae"] = round(metrics["train"]["train_rmae"], 3)
    metrics["train"]["train_mae"] = round(metrics["train"]["train_mae"], 3)
    print(metrics)

    return metrics


if __name__ == "__main__":
    # get arguments
    import argparse

    parser = argparse.ArgumentParser()

    try:
        args = parser.parse_args()
        print(get_metrics())

    except Exception as ex:
        raise (ex)
