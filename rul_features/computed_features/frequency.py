import pyhht
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from numpy import arange, abs
from scipy.fftpack import hilbert
from scipy.signal import stft
from rul_features.computed_features.basic_statistical import *
from rul_features.computed_features.entropy import shannon_entropy, spectral_entropy, permutation_entropy


import pandas as pd

"""
Frequency Features:
 - Statistical frequency features
 - Frequency Spectra
"""


# Statistical frequency features #
def frequency_mean(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return mean(spectrum_df, spectrum_key)


def frequency_maximum(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return maximum(spectrum_df, spectrum_key)


def frequency_minimum(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return minimum(spectrum_df, spectrum_key)


def frequency_root_mean_square(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return root_mean_square(spectrum_df, spectrum_key)


def frequency_peak_to_peak_value(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return peak_to_peak_value(spectrum_df, spectrum_key)


def frequency_variance(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return variance(spectrum_df, spectrum_key)


def frequency_skewness(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return skewness(spectrum_df, spectrum_key)


def frequency_kurtosis(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return kurtosis(spectrum_df, spectrum_key)


def frequency_peak_factor(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return peak_factor(spectrum_df, spectrum_key)


def frequency_change_coefficient(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return change_coefficient(spectrum_df, spectrum_key)


def frequency_clearance_factor(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return clearance_factor(spectrum_df, spectrum_key)


def frequency_abs_energy(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return abs_energy(spectrum_df, spectrum_key)



"""
Entropy features on frequency spectrum
"""


def frequency_permutation_entropy(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return permutation_entropy(spectrum_df, spectrum_key)


def frequency_spectral_entropy(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return spectral_entropy(spectrum_df, spectrum_key)


def frequency_shannon_entropy(current_observation: pd.DataFrame, raw_key: str):
    signal = current_observation[raw_key]
    spectrum_df, spectrum_key = fft_spectrum(signal)

    return shannon_entropy(spectrum_df, spectrum_key)


# Frequency spectra features #
def fft_spectrum(signal: pd.Series, sampling_rate: float = 25600) -> (pd.DataFrame, str):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    Source: https://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
    """
    df_key: str = '|amplitude|'
    n = len(signal)  # length of the signal
    k = arange(n)
    T = n / sampling_rate
    frq = k / T  # two sides frequency range
    frq = frq[range(int(n / 2))]  # one side frequency range

    frequency_values = fft(signal) / n  # fft computing and normalization
    frequency_values = frequency_values[range(int(n / 2))]
    frequency_values = abs(frequency_values)
    return pd.DataFrame(frequency_values, index=frq, columns=[df_key]), df_key


def short_time_fourier_transform(signal: pd.Series, sampling_rate: int = 256000) -> np.array:
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
    :param signal:
    :param sampling_rate:
    :return:
    """
    f, t, Zxx = stft(signal, sampling_rate)
    return abs(Zxx)


def instant_phase(imfs):
    """Extract analytical signal through Hilbert Transform."""
    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    phase = np.unwrap(np.angle(analytic_signal))  # Compute angle between img and real
    return phase


def instant_amplitude(imfs):
    """Extract analytical signal through Hilbert Transform."""
    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    analytic_amplitude = np.abs(analytic_signal)  # Compute amplitude
    return analytic_amplitude


def hht_transform(signal: pd.Series, sampling_rate: int = 256):
    """
    The Hilbert-Huang transform is useful for performing time-frequency analysis of nonstationary and nonlinear data. The Hilbert-Huang procedure consists of the following steps:

    1. emd decomposes the data set x into a finite number of intrinsic mode functions.

    2. For each intrinsic mode function, xi, the function hht:

        Uses hilbert to compute the analytic signal, zi(t)=xi(t)+jH{xi(t)}, where H{xi} is the Hilbert transform of xi.

        Expresses zi as zi(t)=ai(t) ejθi(t), where ai(t) is the instantaneous amplitude and θi(t) is the instantaneous phase.

        Computes the instantaneous energy, ai(t)2, and the instantaneous frequency, ωi(t)≡dθi(t)/dt. If given a sample rate, hht converts ωi(t) to a frequency in Hz.

        Outputs the instantaneous energy in imfinse and the instantaneous frequency in imfinsf.

    When called with no output arguments, hht plots the energy of the signal as a function of time and frequency, with color proportional to amplitude.
    """
    Ts = sampling_rate
    emd_signal = pyhht.EMD(signal)
    imfs = emd_signal.decompose()

    b = []
    d = []
    for imf in imfs:
        b += [np.sum(np.multiply(imf, imf))]
        th = np.angle(hilbert(imf))
        d += [np.diff(th) / Ts / (2 * math.pi)]
    b = np.array(b)
    v = np.argsort(b)

    b_max = np.max(b)
    b = [(1 - x) / b_max for x in b]

    N = len(signal)
    c = np.linspace(0, (N - 2) * Ts, N - 1)
    for k in v[0:1]:
        plt.plot(c, d[k], color=[b[k], b[k], b[k]], markersize=3)
    plt.show()
    pass
