"""
Contains all basic statistical features that can be computed from one observation.
"""
import pandas as pd
import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsf
import math

np.seterr('raise')

"""
Vibration Features:
 - Basic statistical
 - Entropy features
 - frequency features 
"""


# Basic statistical features #
def mean(current_observation: pd.DataFrame, raw_key: str):
    return current_observation[raw_key].mean()


# Feature list taken from Mao et al. 2020
def maximum(current_observation: pd.DataFrame, raw_key: str):
    return current_observation[raw_key].max()


def minimum(current_observation: pd.DataFrame, raw_key: str):
    return current_observation[raw_key].min()


def root_mean_square(current_observation: pd.DataFrame, raw_key: str):
    return math.sqrt(current_observation[raw_key].pow(2).mean())


def abs_avg(current_observation: pd.DataFrame, raw_key: str):
   return root_mean_square(current_observation, raw_key)


def peak_to_peak_value(current_observation: pd.DataFrame, raw_key: str):
    return maximum(current_observation, raw_key) - minimum(current_observation, raw_key)


def standard_deviation(current_observation: pd.DataFrame, raw_key: str):
    return np.std(current_observation[raw_key])


def skewness(current_observation: pd.DataFrame, raw_key: str):
    return tsf.skewness(current_observation[raw_key])


def kurtosis(current_observation: pd.DataFrame, raw_key: str):
    return tsf.kurtosis(current_observation[raw_key])


def variance(current_observation: pd.DataFrame, raw_key: str):
    return tsf.variance(current_observation[raw_key])


def peak_factor(current_observation: pd.DataFrame, raw_key: str):
    root_mean_square_val = root_mean_square(current_observation, raw_key)
    if root_mean_square_val == 0:
        return 0
    return maximum(current_observation, raw_key) / root_mean_square_val


def change_coefficient(current_observation: pd.DataFrame, raw_key: str):
    standard_deviation_val = standard_deviation(current_observation, raw_key)
    if standard_deviation_val == 0:
        return 0
    return mean(current_observation, raw_key) / standard_deviation_val


def clearance_factor(current_observation: pd.DataFrame, raw_key: str):
    mean_val = current_observation[raw_key].pow(2).mean()
    if mean_val == 0:
        return 0 
    return maximum(current_observation, raw_key) / mean_val


def abs_energy(current_observation: pd.DataFrame, raw_key: str):
    return tsf.abs_energy(current_observation[raw_key])


if __name__ == '__main__':
    signal = pd.DataFrame([{1: 1}, {1: 2}, {1: 3}, {1: 4}])