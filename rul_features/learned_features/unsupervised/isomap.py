import pandas as pd
from sklearn.manifold import Isomap

from rul_features.learned_features.Embedder import Embedding


def isomap_embedded_data_frame(bearing_df: pd.DataFrame, verbose: bool = False, n_components: int = 5):
    """
    Embed all observations of a bearig time series using isomap.
    :param bearing_df: Data frame which contains computed features or raw features.
    :return: Isomap embedded data frame.
    """
    isomap = Isomap(n_components=n_components)
    df_transformed = isomap.fit_transform(X=bearing_df)
    if verbose:
        print("Reconstruction error: ", isomap.reconstruction_error())
    return pd.DataFrame(df_transformed), isomap


class IsomapEmbedding(Embedding):
    def fit_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        isomap = Isomap(n_components=encoding_size)
        isomap.fit(X=df)
        self.embedder = isomap

    def embed_data_frame(self, df: pd.DataFrame):
        return pd.DataFrame(self.embedder.transform(df))
