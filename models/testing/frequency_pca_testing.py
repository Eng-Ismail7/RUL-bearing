from rul_features.learned_features.unsupervised.principal_component_analysis import PCAEmbedding
from models.CombinedFeaturesFFNN import EmbeddingFeaturesFNNN, DataSetType
from rul_features.computed_features.frequency import fft_spectrum
from pre_processing.raw_features import read_raw_dfs_as_dict
from util.constants import LEARNING_SET
from util.helper import concat_dfs
import pandas as pd


def freuqency_pca_embedding_test():
    # Read Training data
    pca_training_data = read_raw_dfs_as_dict(LEARNING_SET)
    # Remove labels
    pca_training_data = concat_dfs(pca_training_data)
    labels = pca_training_data.pop('RUL')
    embedder = PCAEmbedding()
    model = EmbeddingFeaturesFNNN(name="Some_debugging", embedding_method=embedder,
                                  encoding_size=450, data_set_type=DataSetType.raw, use_frequency_embedding=True)
    pca_training_data = pca_training_data.apply(lambda x: extract_amplitude_column(x), axis=1, result_type="expand")

def extract_amplitude_column(signal: pd.Series):
    spec_df, df_key = fft_spectrum(signal)
    return spec_df[df_key]

if __name__ == '__main__':
    freuqency_pca_embedding_test()
