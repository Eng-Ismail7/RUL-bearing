from pre_processing.features import read_feature_dfs_as_dict
from pre_processing.features import df_dict_to_df_dataframe
from util.helper import pop_labels
from models.CombinedFeaturesFFNN import EmbeddingFeaturesFNNN
from rul_features.learned_features.unsupervised.isomap import IsomapEmbedding

from util.metrics import rmse, correlation_coefficient
from util.constants import LEARNING_SET, FULL_TEST_SET, ENTROPY_FEATURES, ALL_FEATURES
from util.logging import save_latex_aggregated_table
from models.DataSetType import DataSetType


def eval_isomap():
    training_data, training_labels = df_dict_to_df_dataframe(read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET))
    validation_dict = read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET)
    validation_labels = pop_labels(validation_dict)

    isomap_5 = EmbeddingFeaturesFNNN(name="Isomap combined 5",
                                     embedding_method=IsomapEmbedding(),
                                     encoding_size=5,
                                     data_set_type=DataSetType.computed)
    isomap_15 = EmbeddingFeaturesFNNN(name="Isomap combined 15",
                                      embedding_method=IsomapEmbedding(),
                                      encoding_size=15,
                                      data_set_type=DataSetType.computed)
    isomap_25 = EmbeddingFeaturesFNNN(name="Isomap combined 25",
                                      embedding_method=IsomapEmbedding(),
                                      encoding_size=25,
                                      data_set_type=DataSetType.computed)
    isomap_35 = EmbeddingFeaturesFNNN(name="Isomap combined 35",
                                      embedding_method=IsomapEmbedding(),
                                      encoding_size=35,
                                      data_set_type=DataSetType.computed)
    isomap_45 = EmbeddingFeaturesFNNN(name="Isomap combined 45",
                                      embedding_method=IsomapEmbedding(),
                                      encoding_size=45,
                                      data_set_type=DataSetType.computed)
    isomap_55 = EmbeddingFeaturesFNNN(name="Isomap combined 55",
                                      embedding_method=IsomapEmbedding(),
                                      encoding_size=55,
                                      data_set_type=DataSetType.computed)

    isomap_models = [isomap_5, isomap_15, isomap_25, isomap_35, isomap_45, isomap_55]
    metrics_dict = {}
    for isomap_model in isomap_models:
        # print("Currently evaluating: ", isomap_model.name)
        isomap_model.train(training_data, training_labels, validation_data=None,
                           validation_labels=None)
        metrics_dict[isomap_model.name] = isomap_model.compute_metrics(
            df_dict=validation_dict, labels=validation_labels,
            metrics_list=[rmse, correlation_coefficient])
    save_latex_aggregated_table(metrics_dict, None)


if __name__ == '__main__':
    eval_isomap()
