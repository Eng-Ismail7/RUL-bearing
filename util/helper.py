import pandas as pd
from typing import Dict
from util.constants import BEARING_COLUMN


def pop_labels(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    return {bearing: df.pop('RUL') for bearing, df in df_dict.items()}


def concat_dfs(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    assert len(df_dict) > 0
    # assert "RUL" in df_dict[df_dict.keys()[0]].keys()
    concatenated_dfs = pd.concat(df_dict, keys=df_dict.keys(), names=["bearing", "index"])
    return concatenated_dfs


def reformat_flattened_data(df_list: list, n_rows: int, n_cols: int) -> list:
    return [bearing.to_numpy().reshape((bearing.shape[0], n_rows, n_cols)) for bearing in df_list]


def flatten_predictions(prediction_list: list) -> list:
    return [prediction[0] for prediction in prediction_list]
