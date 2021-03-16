"""
This file contains constant variables like path names and feature sets.
"""
from rul_features.computed_features.basic_statistical import *
from rul_features.computed_features.entropy import *
from rul_features.computed_features.frequency import *


# # Data Set File Paths
# DATA_SET_PATH = \
#     'C:\\Your\\Local\\Path\\To\\Original\\Bearing\\Data'
# PROCESSED_DATA_SET_PATH = \
#     'C:\\Your\\Local\\Path\\To\\Processed\\Bearing\\Data'

# # Metrics Dicts File Path
# METRICS_DICT_PATH = \
#     "C:\\Your\\Local\\Path\\To\\Repository\\metrics_dict"

# # Memory Optimization Cache
# MEMORY_CACHE_PATH = \
#     'C:\\Your\\Local\\Path\\To\\Repository\\memory_cache'

# # CNN Path
# CNN_PATH = \
#     "C:\\Your\\Local\\Path\\To\\Repository\\cnn_embedding_layers"

# Data Set File Paths
DATA_SET_PATH = \
    'C:\\Users\\Administrator\\Documents\\_GIT\\_IFIS\\rul_clean\\remaining-useful-lifetime\\data\\raw'
PROCESSED_DATA_SET_PATH = \
    'C:\\Users\\Administrator\\Documents\\_GIT\\_IFIS\\rul_clean\\remaining-useful-lifetime\\data\\processed'

# Metrics Dicts File Path
METRICS_DICT_PATH = \
    "C:\\Users\\Administrator\\Documents\\_GIT\\_IFIS\\rul_clean\\remaining-useful-lifetime\\metrics_dict"

# Memory Optimization Cache
MEMORY_CACHE_PATH = \
    'C:\\Users\\Administrator\\Documents\\_GIT\\_IFIS\\rul_clean\\remaining-useful-lifetime\\memory_cache'

# CNN Path
CNN_PATH = \
    "C:\\Users\\Administrator\\Documents\\_GIT\\_IFIS\\rul_clean\\remaining-useful-lifetime\\cnn_embedding_layers"

# File names
FEATURES_CSV_NAME = 'features.csv'
SPECTRA_CSV_NAME = 'spectra.csv'
RAW_CSV_NAME = 'acc.csv'

# data set_sub_set
FULL_TEST_SET = 'Full_Test_Set'
LEARNING_SET = 'Learning_set'
TEST_SET = 'Test_set'

# Pandas keys
RUL_KEY = 'RUL'
STACKED_SPECTRA_KEY = 'stacked_spectra'
SPECTRA_SHAPE_KEY = 'spectra_shape'
HORIZONTAL_SPECTRA_KEY = 'h_spectra'
VERTICAL_SPECTRA_KEY = 'v_spectra'
BEARING_COLUMN = "bearing_name"

# Metric keys
RMSE_KEY = "RMSE"
CORR_COEFF_KEY = "PCC"
STANDARD_DEVIATION_KEY = "STD"

# Facts
SAMPLING_FREQUENCY = 25600
SPECTRA_SHAPE = (129, 21, 2)

# Feature sets
OLD_STATISTICAL_FEATURES = [mean, maximum, minimum, root_mean_square, abs_avg, peak_to_peak_value, standard_deviation,
                            skewness, kurtosis, variance, peak_factor, change_coefficient, clearance_factor,
                            abs_energy]

BASIC_STATISTICAL_FEATURES = [mean, maximum, minimum, root_mean_square, peak_to_peak_value, skewness, kurtosis,
                              variance, peak_factor, change_coefficient, clearance_factor, abs_energy]

ENTROPY_FEATURES = [shannon_entropy, permutation_entropy]

ENTROPY_FREQUENCY_FEATURES = [frequency_shannon_entropy, frequency_permutation_entropy]

FREQUENCY_FEATURES = [frequency_mean, frequency_maximum, frequency_minimum, frequency_root_mean_square,
                      frequency_peak_to_peak_value, frequency_skewness, frequency_kurtosis, frequency_variance,
                      frequency_peak_factor, frequency_change_coefficient, frequency_clearance_factor,
                      frequency_abs_energy]

FREQUENCY_ENTROPY_FEATURES = [spectral_entropy]

ALL_FEATURES = BASIC_STATISTICAL_FEATURES + ENTROPY_FEATURES + FREQUENCY_FEATURES + ENTROPY_FREQUENCY_FEATURES
