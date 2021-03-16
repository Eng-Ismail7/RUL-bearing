import os
import glob
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Sequence, List
from util.constants import *
import sys


def concatenate_raw_features(sub_set: str):
    """
    Concatenate all raw feature observations of all bearings and store them in a csv file.
    :param sub_set: data set sub set in {LEARNING_SET, TEST_SET, FULL_TEST_SET}
    :return: Void, Write CSVs to file_system
    """
    data_set_subset_path_in = Path(DATA_SET_PATH).joinpath(sub_set)
    data_set_subset_path_out = Path(PROCESSED_DATA_SET_PATH).joinpath(sub_set)
    bearings_list = os.listdir(data_set_subset_path_in)
    """
    Specify the columns that should be used from the raw data.
    """
    types_infos = {
        'acc': {'usecols': [0, 1, 2, 3, 4, 5], 'names': ['hour', 'min', 's', 'seg', 'h', 'v']}  # ,
    }

    print("Concatenating raw features, for %d bearings" % len(bearings_list))
    for bearing in bearings_list:
        path_in = data_set_subset_path_in.joinpath(bearing)
        path_out = data_set_subset_path_out.joinpath(bearing)
        csvs_merge(path_in=path_in, path_out=path_out, types_infos=types_infos)


def csvs_merge(path_in, path_out, types_infos):
    """
    Merge all .csv files of a 'file_type' in 'path_in'
    The merged file is saved in "data/processed_data/'dataset'/'folder'/merged_files.csv"
    adapted from https://github.com/matheuscnali/bearing_rul_predict
    """

    for file_type, type_infos in types_infos.items():

        all_observations = read_raw_observations(path_in, file_type, types_infos)

        for i in range(0, len(all_observations)):
            all_observations[i]['#observation'] = i
        merged_observations = pd.concat(all_observations)

        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        merged_observations.to_csv("%s/%s.csv" % (path_out, file_type), index=False, encoding='utf-8-sig')


def read_raw_observations(bearing_path_in, file_type, type_infos) -> List[pd.DataFrame]:
    all_observations = []
    # Getting all .csv files of type 'file_type'
    csv_files = sorted(glob.glob('%s/%s*.csv' % (bearing_path_in, file_type)))
    # Some folders don't have all types
    if not csv_files:
        return all_observations

    # Determining separator type with the first .csv file
    reader = pd.read_csv(csv_files[0], sep=None, iterator=True, engine='python')
    inferred_sep = reader._engine.data.dialect.delimiter

    # Merging .csv files of type 'file_type'
    for file in csv_files:
        current_observation = pd.read_csv('%s' % file,
                                          sep=inferred_sep,
                                          usecols=type_infos['usecols'],
                                          names=type_infos['names'],
                                          header=None, engine='c')
        all_observations = all_observations + [current_observation]
    return all_observations


def read_raw_dfs_as_dict(sub_set: str) -> Dict[str, pd.DataFrame]:
    data_set_subset_path_in = Path(DATA_SET_PATH).joinpath(sub_set)
    bearings_list = os.listdir(data_set_subset_path_in)
    """
    Specify the columns that should be used from the raw data.
    """
    types_infos = {
        'acc': {'usecols': [4, 5], 'names': ['h', 'v']}  # ,
    }
    all_bearings = {}
    bearings_list: Sequence[str] = sorted(bearings_list)
    for bearing in tqdm(bearings_list, desc="Reading raw data of bearings."):
        path_in = data_set_subset_path_in.joinpath(bearing)
        for file_type, type_infos in types_infos.items():
            curr_bearing = read_raw_observations(bearing_path_in=path_in, file_type=file_type, type_infos=type_infos)
            curr_bearing = [observation.transpose() for observation in curr_bearing]
            curr_bearing = pd.concat(curr_bearing, ignore_index=True)
            add_rul(curr_bearing)
            all_bearings[bearing] = curr_bearing
    return all_bearings


def add_rul(df: pd.DataFrame):
    df['RUL'] = 0
    amount_observations = df.shape[0]
    for i in range(amount_observations):
        df.at[i, 'RUL'] = (amount_observations - i) * 10
