import os
from typing import Callable
from pathlib import Path
from util.constants import *
from rul_features.computed_features.frequency import short_time_fourier_transform
from tqdm import tqdm
from pre_processing.raw_features import read_raw_observations


def compute_spectra_all_bearings(sub_set: str):
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
    print("Computing spectra, for %d bearings" % len(bearings_list))
    for i in range(len(bearings_list)):
        bearing = bearings_list[i]
        path_in = data_set_subset_path_in.joinpath(bearing)
        path_out = data_set_subset_path_out.joinpath(bearing)
        compute_spectra(path_in=path_in, path_out=path_out, types_infos=types_infos, bearing_num=i + 1)


def compute_spectra(path_in, path_out, types_infos,
                    frequency_function: Callable[[pd.Series, int], np.array] = short_time_fourier_transform,
                    bearing_num: int = 0):
    """
    :param path_in: Raw data set input path.
    :param path_out: Output path where processed data is stored as a csv
    :param types_infos: Specifies used columns of raw data
    :param frequency_function: fnuction used to determine time-frequency spectrum
    :param bearing_num: helper param for progres bar description
    :return: Void, writes computed features into file system
    """
    for file_type, type_infos in types_infos.items():
        all_observations = read_raw_observations(path_in, file_type, type_infos)
        all_spectra_features = []
        with tqdm(range(len(all_observations)), position=0, leave=True) as t:
            for i in t:
                t.set_description('Computing time-frequency spectra for Bearing: %d' % bearing_num)
                current_observation = all_observations[i]
                horizontal_spectra = frequency_function(current_observation['h'], SAMPLING_FREQUENCY)
                vertical_spectra = frequency_function(current_observation['v'], SAMPLING_FREQUENCY)
                stacked_spectra = np.dstack([horizontal_spectra, vertical_spectra])
                all_spectra_features += [stacked_spectra.flatten()]

            spectra_features = pd.DataFrame(data=all_spectra_features)
            # Add RUL label
            amount_observations = spectra_features.shape[0]
            spectra_features[RUL_KEY] = 0
            for i in range(amount_observations):
                spectra_features.at[i, RUL_KEY] = (amount_observations - i) * 10
            # Create processed data set directory
            if not os.path.exists(path_out):
                Path(path_out).mkdir(parents=True, exist_ok=True)
            print("Writing spectra data of bearing %d as CSV." % bearing_num)
            spectra_features.to_csv("%s/%s" % (path_out, SPECTRA_CSV_NAME), index=False, encoding='utf-8-sig')
