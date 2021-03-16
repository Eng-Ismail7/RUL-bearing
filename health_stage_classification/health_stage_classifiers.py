import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import sys
"""
Implementations of two stage classifications methods as proposed in:
 - Li et al. 2019 2*sigma interval of kurtosis
 - Ahmad et al. 2019 Alarm bound technique
 - Mao et al. 2018 SVD normalized correlation coefficient
"""


def li_et_al_2019(kurtosis: pd.Series, normal_period: range = range(50, 150), sigma_interval: float = 2) -> int:
    """
    Li et al. in 2019 used the kurtosis as a classification indicator by computing its mean and standard deviation
    in the early period of bearing operation. The first prediction time (FPT )was then determined as the point in time
    where the kurtosis exceeds the 2*std_dev interval.

    :param kurtosis: kurtosis that is used as the FPT indicator
    :param normal_period: Range of the period that is representative for normal bearing behaviour.
    :param sigma_interval: range of deviation that is allowed for the kurtosis
    :return: index of FPT
    """

    kurtosis_normal = kurtosis[normal_period]
    mean = kurtosis_normal.mean()
    std_dev = kurtosis_normal.std()

    kurtosis = kurtosis - mean
    kurtosis = kurtosis.abs()

    kurtosis = np.array(kurtosis)
    n = kurtosis.size
    threshold = sigma_interval * std_dev
    for i in range(150, n):
        if kurtosis[i - 1] > threshold:
            if kurtosis[i] > threshold:
                return i
    return 0


def ahmad_et_al_2019(root_mean_square: pd.Series, window_size: int = 70) -> int:
    rms_normal = root_mean_square.iloc[0:100].mean()
    health_indicator = pd.Series(root_mean_square / rms_normal, name='ahmad_health_indicator')
    lrt: pd.Series = linear_rectification_technique(health_indicator)

    if window_size > len(lrt):
        raise Exception("Window size is longer than the health indicator signal.")
    for i in range(len(lrt) - window_size):
        start_index = i
        end_index = i + window_size
        value_range = lrt.iloc[start_index:end_index]
        index = [[x] for x in range(0, window_size)]
        values = [[y] for y in value_range]
        lin_reg = LinearRegression().fit(X=index, y=values)
        gain = abs(lin_reg.coef_[0][0] * window_size)
        if gain / lin_reg.intercept_ > 0.1:
            return end_index
    return 0


def linear_rectification_technique(signal: pd.Series) -> pd.Series:
    n: int = len(signal)
    growth_rate = signal.diff(1).mean()
    smoothed = [signal.iloc[0]]
    for i in range(1, n):
        h_i = signal.iloc[i]
        h_i_min = smoothed[i - 1]
        if h_i_min <= h_i <= (1 + growth_rate) * h_i_min:
            smoothed += [h_i]
        elif (h_i < h_i_min) or (h_i > (1 + growth_rate) * h_i_min):
            smoothed += [h_i_min + growth_rate]
    return pd.Series(smoothed, name='lrt')


def cut_fpts(df_dict: Dict[str, pd.DataFrame], fpt_method=li_et_al_2019, signal_key: str = 'kurtosis') -> (
        Dict[str, pd.DataFrame], Dict[str, int]):
    """
    Cuts data frames so they only include the unhealthy stage of a bearing.
    :param df_dict: dict with data frames that the first prediction time should be computed for and that will be cut
    :param fpt_method: method by which the first prediction time will be computed.
    :return: list of integers with the first prediction times, list of shortened data frames
    """
    first_prediction_times = {}
    cut_dfs = {}
    for bearing, df in df_dict.items():
        fpt: int = fpt_method(df[signal_key])
        first_prediction_times[bearing] = fpt
        cut_dfs[bearing] = df_dict[bearing][fpt:]
    return cut_dfs, first_prediction_times


def procentual_rul(df_list: list, fpts: list):
    for i in tqdm(range(len(df_list))):
        rul = df_list[i].pop('RUL')
        lifetime = rul.max()
        p_rul = [(r * 100) / (lifetime - fpts[i]) for r in rul]
        df_list[i].at['RUL'] = pd.Series(p_rul, index=df_list[i].index)
    return df_list


import matplotlib.pyplot as plt
from scipy import stats


def random_line(m, b, sigma, size=10):
    np.random.seed(123)
    xdata = np.linspace(-1.0, 1.0, size)
    # Generate normally distributed random error ~ N(0, sigma**2)
    errors = stats.norm.rvs(loc=0, scale=sigma, size=size)
    ydata = m * xdata + b + errors
    return xdata, ydata


if __name__ == '__main__':
    siz = 100
    _, line = random_line(50, 5000, 30, size=siz)
    plt.plot(range(siz), line)
    plt.plot(range(siz), linear_rectification_technique(pd.Series(line)))
    plt.show()
