import pandas as pd
import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsf
from rul_features.computed_features.basic_statistical import peak_to_peak_value
from pyts.approximation import SymbolicAggregateApproximation
from scipy.stats import entropy
from pyhht.emd import EMD

"""
Vibration Features:
 - Basic statistical
 - Entropy features
 - frequency features 
"""


# Entropy features #
def permutation_entropy(current_observation: pd.DataFrame, raw_key: str = 'h'):
    # input, tau (step), dim (window)
    return tsf.permutation_entropy(current_observation[raw_key], 10, 5)


def sample_entropy(current_observation: pd.DataFrame, raw_key: str):
    return tsf.sample_entropy(current_observation[raw_key])


def approximate_entropy(current_observation: pd.DataFrame, raw_key: str):
    # m Length of compared run of data
    # r filtering level
    return tsf.approximate_entropy(current_observation[raw_key], 3, peak_to_peak_value(current_observation, raw_key))


def binned_entropy(current_observation: pd.DataFrame, raw_key: str):
    return tsf.binned_entropy(current_observation[raw_key], 1280)


def spectral_entropy(current_observation: pd.DataFrame, raw_key: str):
    return tsf.fourier_entropy(current_observation[raw_key], 1280)


def shannon_entropy(current_observation: pd.DataFrame, raw_key: str):
    result = None
    try:
        sax = SymbolicAggregateApproximation()
        time_series = [current_observation[raw_key].to_numpy()]

        symbolic_representation = sax.transform(time_series)
        _, counts = np.unique(symbolic_representation, return_counts=True)
        frequencies = counts / len(symbolic_representation[0])
        result = entropy(pk=frequencies)
    except ValueError as e:
        message = str(e)
        if message == 'At least one sample is constant.':
            result = 0

    return result


if __name__ == '__main__':
    uniform = pd.DataFrame(pd.Series(np.array([1, 1, 1, 1, 2]), name='h'))