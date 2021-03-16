from rul_features.learned_features.unsupervised.autoencoder import fit_autoencoder
from pre_processing.raw_features import read_raw_dfs
from util.constants import LEARNING_SET
from util.helper import concat_dfs


def autoencoder_features_no_classifier_ffnn_rul_prediction():
    """
    VERY VERY Bad -> autoencoder hyperparams?
    :return: Void
    """
    # Read Training data
    autoencoder_training_data = read_raw_dfs(sub_set=LEARNING_SET)

    # Remove labels
    autoencoder_training_data = concat_dfs(autoencoder_training_data)
    labels = autoencoder_training_data.pop('RUL')
    autoencoder, encoder = fit_autoencoder(autoencoder_training_data, epochs=100, encoding_size=80)

if __name__ == '__main__':
    autoencoder_features_no_classifier_ffnn_rul_prediction()
