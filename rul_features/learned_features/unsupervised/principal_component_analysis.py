import pandas as pd
from sklearn.decomposition import PCA
from rul_features.learned_features.Embedder import Embedding
import sys

def pca_embedded_data_frame(df: pd.DataFrame, n_components=0.9, verbose: bool = False):
    """
    Embed all observations of a bearing time series using PCA.
    :param df: Data frame which contains computed features or raw features.
    :param n_components: Number of components to be learned from PCA
    :return: PCA embedded data frame.
    """
    normalized_df = (df-df.mean())/df.std()
    pca = PCA(n_components=n_components)
    df_transformed = pca.fit_transform(X=normalized_df)
    if verbose:
        print("Original DF: ", normalized_df.head())
        print("Reconstructed DF: ", pd.DataFrame(pca.inverse_transform(df_transformed)).head())

    return pd.DataFrame(df_transformed), pca


class PCAEmbedding(Embedding):
    def fit_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        pca = PCA(n_components=encoding_size)
        pca.fit(X=df)
        self.embedder = pca

    def embed_data_frame(self, df: pd.DataFrame):
        return pd.DataFrame(self.embedder.transform(df))
