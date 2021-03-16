from pre_processing.features import read_feature_dfs_as_dict
from pre_processing.features import df_dict_to_df_dataframe
from util.helper import pop_labels
from models.ComputedFeaturesFFNN import ComputedFeaturesFFNN
from util.metrics import rmse, correlation_coefficient
from util.constants import LEARNING_SET, FULL_TEST_SET, ENTROPY_FEATURES, ALL_FEATURES
from util.logging import save_latex_aggregated_table
from util.visualization import plot_raw_features

def train_svr():
    training_data, training_labels = df_dict_to_df_dataframe(read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET))
    validation_dict = read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET)
    validation_labels = pop_labels(validation_dict)

    svr_model = ComputedFeaturesFFNN(name="SVR", feature_list=ENTROPY_FEATURES)

    svr_model.train_svr(training_data, training_labels)
    metrics_dict = {"Entropy Poly": svr_model.compute_metrics(df_dict=validation_dict, labels=validation_labels,
                                                              metrics_list=[rmse, correlation_coefficient],
                                                              use_svr=True)}
    svr_model.visualize_rul(df_dict=validation_dict, label_data=validation_labels, use_svr=True, experiment_name=None)
    save_latex_aggregated_table(metrics_dict, None)



if __name__ == '__main__':
    train_svr()
