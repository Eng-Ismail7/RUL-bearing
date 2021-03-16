from pre_processing.raw_features import read_raw_dfs_as_dict
from pre_processing.features import df_dict_to_df_dataframe, read_feature_dfs_as_dict
from util.helper import pop_labels
from models.CombinedFeaturesFFNN import EmbeddingFeaturesFNNN
from rul_features.learned_features.unsupervised.principal_component_analysis import PCAEmbedding

from util.metrics import rmse, correlation_coefficient
from util.constants import LEARNING_SET, FULL_TEST_SET, ENTROPY_FEATURES, ALL_FEATURES
from util.constants import RAW_CSV_NAME
from util.logging import save_latex_aggregated_table
from models.DataSetType import DataSetType


def train_pca():
    training_data, training_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET, csv_name=RAW_CSV_NAME))
    validation_dict = read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET, csv_name=RAW_CSV_NAME)
    validation_labels = pop_labels(validation_dict)

    computed_features_pca_combiner_ffnn = EmbeddingFeaturesFNNN(name="Isomap combined",
                                                                embedding_method=PCAEmbedding(),
                                                                encoding_size=5,
                                                                data_set_type=DataSetType.computed)

    computed_features_pca_combiner_ffnn.train(training_data, training_labels, validation_data=None,
                                              validation_labels=None)
    metrics_dict = {computed_features_pca_combiner_ffnn.name: computed_features_pca_combiner_ffnn.compute_metrics(
        df_dict=validation_dict, labels=validation_labels,
        metrics_list=[rmse, correlation_coefficient])}
    computed_features_pca_combiner_ffnn.visualize_rul(df_dict=validation_dict,
                                                      label_data=validation_labels, experiment_name=None)
    save_latex_aggregated_table(metrics_dict, None)


def train_raw_pca():
    training_data, training_labels = df_dict_to_df_dataframe(read_raw_dfs_as_dict(sub_set=LEARNING_SET))

    computed_features_pca_combiner_ffnn = EmbeddingFeaturesFNNN(name="PCA learned",
                                                                embedding_method=PCAEmbedding(),
                                                                encoding_size=1280,
                                                                data_set_type=DataSetType.computed)

    computed_features_pca_combiner_ffnn.train(training_data, training_labels, validation_data=None,
                                              validation_labels=None)

if __name__ == '__main__':
    train_raw_pca()
