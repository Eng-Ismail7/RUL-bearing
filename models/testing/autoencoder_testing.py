from pre_processing.features import read_feature_dfs_as_dict, df_dict_to_df_dataframe
from pre_processing.raw_features import read_raw_dfs_as_dict
from rul_features.learned_features.unsupervised.autoencoder import fit_autoencoder

from util.metrics import rmse, correlation_coefficient
from util.constants import LEARNING_SET, FULL_TEST_SET, ENTROPY_FEATURES, ALL_FEATURES


def train_autoencoder():
    training_data, training_labels = df_dict_to_df_dataframe(read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET))
    validation_dict, validation_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET))

    _, _ = fit_autoencoder(training_data, encoding_size=25, validation_data=(validation_dict, validation_labels))


def train_raw_autoencoder():
    training_data, training_labels = df_dict_to_df_dataframe(read_raw_dfs_as_dict(sub_set=LEARNING_SET))
    validation_dict, validation_labels = df_dict_to_df_dataframe(
        read_raw_dfs_as_dict(sub_set=FULL_TEST_SET))

    _, _ = fit_autoencoder(training_data, encoding_size=450, validation_data=(validation_dict, validation_labels))


if __name__ == '__main__':
    train_raw_autoencoder()
