import abc
import pandas as pd

from rul_features.computed_features.frequency import fft_spectrum


def _extract_amplitude_column(signal: pd.Series):
    spec_df, df_key = fft_spectrum(signal)
    return spec_df[df_key]

class Embedding(abc.ABC):
    def __init__(self):
        self.embedder = None

    @abc.abstractmethod
    def fit_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        pass

    @abc.abstractmethod
    def embed_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_frequency_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        """

        :param df: df containing all observations across all bearing without their RUL
        :param encoding_size:
        :param verbose:
        :return:
        """
        df = df.apply(lambda x: _extract_amplitude_column(x), axis=1, result_type="expand")
        self.fit_embedding(df, encoding_size, verbose)

    def embed_frequency_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(lambda x: _extract_amplitude_column(x), axis=1, result_type="expand")
        return self.embed_data_frame(df)
