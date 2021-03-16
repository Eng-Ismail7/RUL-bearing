from pre_processing.features import read_feature_dfs

from util.constants import LEARNING_SET, FULL_TEST_SET, BASIC_STATISTICAL_FEATURES
from util.visualization import plot_trainings_history, plot_rul_comparisons, plot_fpts
from util.helper import *

from rul_prediction.ffnn import fit_ffnn
from health_stage_classification.health_stage_classifiers import ahmad_et_al_2019, cut_fpts


def tba_features_ahmad_et_al_2019_ffnn():
    feature_list = BASIC_STATISTICAL_FEATURES

    fpt_method = ahmad_et_al_2019
    signal_key = "root_mean_square"
    learning_set = read_feature_dfs(data_set_sub_set=LEARNING_SET, features=feature_list)

    # Calculate FPTs and cut dataframes
    cut_learning_set, first_prediction_times = cut_fpts(learning_set, fpt_method=fpt_method,
                                                        signal_key=signal_key)

    plot_fpts(first_prediction_times=first_prediction_times, df_list=learning_set,
              classification_indicator="root_mean_square")

    cut_learning_set = concat_dfs(cut_learning_set)
    training_labes = cut_learning_set.pop('RUL')
    ffnn, trainings_history = fit_ffnn(X=cut_learning_set, y=training_labes, epochs=100)
    # Plot trainings history
    plot_trainings_history(trainings_history=trainings_history, error_type='RMSE')

    # Read test set
    testing_set = read_feature_dfs(data_set_sub_set=FULL_TEST_SET, features=feature_list)
    cut_testing_set, _ = cut_fpts(testing_set, fpt_method=fpt_method, signal_key=signal_key)
    testing_labels = pop_labels(cut_testing_set)

    plot_rul_comparisons(cut_testing_set, testing_labels, ffnn)


if __name__ == '__main__':
    tba_features_ahmad_et_al_2019_ffnn()
