import os
from pathlib import Path
from util.constants import *
from typing import Sequence, Dict, Callable
from tqdm import tqdm
from pre_processing.raw_features import read_raw_observations, add_rul
from rul_features.computed_features.frequency import fft_spectrum
from util.helper import concat_dfs


def compute_feature_data_frame(sub_set: str):
    """
    Compute all features of all bearings and store them as a CSV
    :param sub_set: data set sub set in {LEARNING_SET, TEST_SET, FULL_TEST_SET}
    :return: Void, Write CSVs to file_system
    """
    feature_list = ALL_FEATURES
    data_set_subset_path_in = Path(DATA_SET_PATH).joinpath(sub_set)
    data_set_subset_path_out = Path(PROCESSED_DATA_SET_PATH).joinpath(sub_set)
    bearings_list = os.listdir(data_set_subset_path_in)
    """
    Specify the columns that should be used from the raw data.
    """
    types_infos = {
        'acc': {'usecols': [0, 1, 2, 3, 4, 5], 'names': ['hour', 'min', 's', 'seg', 'h', 'v']}
    }
    print("Computing %d features, for %d bearings" % (len(feature_list), len(bearings_list)))
    for i in range(len(bearings_list)):
        bearing = bearings_list[i]
        path_in = data_set_subset_path_in.joinpath(bearing)
        path_out = data_set_subset_path_out.joinpath(bearing)
        compute_csv_features(path_in=path_in, path_out=path_out,
                             types_infos=types_infos, feature_list=feature_list, bearing_num=i + 1)


def compute_csv_features(path_in, path_out, types_infos, feature_list: list, bearing_num: int = 0):
    """
    :param path_in: Raw data set input path.
    :param path_out: Output path where processed data is stored as a csv
    :param types_infos: Specifies used columns of raw data
    :param feature_list: dict that includes list of functions that compute features per observation
    :return: Void, writes computed features into file system
    """
    for file_type, type_infos in types_infos.items():

        all_observations = read_raw_observations(path_in, file_type, type_infos)
        all_observation_features = []
        with tqdm(range(len(all_observations)), position=0, leave=True) as t:
            for i in t:
                t.set_description('Computing features for Bearing: %d' % bearing_num)
                current_observation = all_observations[i]
                all_observation_features = all_observation_features + [
                    compute_features_from_observation(current_observation,
                                                      feature_list, t)]

        # Merge the computed features to one bearing data frame
        merged_features = pd.DataFrame(all_observation_features)
        # Add RUL label
        add_rul(merged_features)

        # Create processed data set directory
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        # Store as .csv
        merged_features.to_csv("%s/%s" % (path_out, FEATURES_CSV_NAME), index=False, encoding='utf-8-sig')


def compute_features_from_observation(current_observation: pd.DataFrame, feature_functions: list, pbar: tqdm) -> Dict[
    str, float]:
    """
    Helper function that computes each feature per observation.
    :param current_observation: observation data frame (len=2560)
    :param feature_functions: Functions that compute a feature out of the current observation.
    :param pbar: tqdm progress bar which postfix is changed depending on the feature being computed
    :return: List of computed features.
    """
    features = {}
    for feature_function in feature_functions:
        pbar.set_postfix({'feature': feature_function.__name__})
        features['h_' + feature_function.__name__] = feature_function(current_observation, 'h')
        features['v_' + feature_function.__name__] = feature_function(current_observation, 'v')
    return features

def read_feature_dfs_as_dict(data_set_sub_set: str, csv_name=FEATURES_CSV_NAME) -> \
        Dict[str, pd.DataFrame]:
    """
    Reads all CSVs and compiles them into a data frame of a given sub set and CSV type
    :param data_set_sub_set: {FULL_TEST_SET, LEARNING_SET, TEST_SET} from constants.py
    :param csv_name: type of features that are to be read
    :return: list of read and compiled data frames
    """
    path = Path(PROCESSED_DATA_SET_PATH).joinpath(data_set_sub_set)
    bearing_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    bearing_list = sorted(bearing_list)
    df_dict = {}
    for bearing in tqdm(bearing_list, desc="Reading computed features of bearings from: %s" % csv_name):
        df_dict[bearing] = pd.read_csv(Path.joinpath(path, bearing, csv_name))
    return df_dict


def read_feature_dfs_as_dataframe(data_set_sub_set: str, csv_name=FEATURES_CSV_NAME) -> (pd.DataFrame, pd.Series):
    """
    Reads all CSVs and compiles them into a data frame of a given sub set and CSV type
    :param data_set_sub_set: {FULL_TEST_SET, LEARNING_SET, TEST_SET} from constants.py
    :param features: type of features that are to be read
    :return: list of read and compiled data frames
    """
    df_dict = read_feature_dfs_as_dict(data_set_sub_set=data_set_sub_set, csv_name=csv_name)
    return df_dict_to_df_dataframe(df_dict)


def df_dict_to_df_dataframe(df_dict: Dict[str, pd.DataFrame]) -> (pd.DataFrame, pd.Series):
    data = concat_dfs(df_dict)
    labels = data.pop("RUL")
    return data, labels
