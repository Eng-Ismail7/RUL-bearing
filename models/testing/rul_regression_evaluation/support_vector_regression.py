from pre_processing.features import read_feature_dfs

from util.constants import LEARNING_SET, FULL_TEST_SET
from util.visualization import plot_trainings_history, plot_rul_comparisons, plot_fpts
from util.helper import *

from rul_prediction.suport_vector_regression import fit_svr
from health_stage_classification.health_stage_classifiers import li_et_al_2019, cut_fpts


def tba_features_li_et_al_2019_svr():
    feature_list = [] 
    use_hs_classifier = True
    fpt_method = li_et_al_2019
    signal_key = "kurtosis"

    learning_set = read_feature_dfs(data_set_sub_set=LEARNING_SET, features=feature_list)

    if use_hs_classifier:
        # Calculate FPTs and cut dataframes
        cut_learning_set, first_prediction_times = cut_fpts(learning_set, fpt_method=fpt_method,
                                                            signal_key=signal_key)

        plot_fpts(first_prediction_times=first_prediction_times, df_list=learning_set,
                  classification_indicator="root_mean_square")

        cut_learning_set = concat_dfs(cut_learning_set)
        training_labes = cut_learning_set.pop('RUL')
        svr, trainings_history = fit_svr(X=cut_learning_set, y=training_labes)
    else:
        learning_set = concat_dfs(learning_set)
        training_labes = learning_set.pop('RUL')
        svr, trainings_history = fit_svr(X=learning_set, y=training_labes)

    # Plot trainings history
    plot_trainings_history(trainings_history=trainings_history, error_type='RMSE')

    # Read test set
    testing_set = read_feature_dfs(data_set_sub_set=FULL_TEST_SET, features=feature_list)
    if use_hs_classifier:
        testing_set, _ = cut_fpts(testing_set, fpt_method=fpt_method, signal_key=signal_key)
    testing_labels = pop_labels(testing_set)

    plot_rul_comparisons(testing_set, testing_labels, svr)
