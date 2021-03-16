from pre_processing.features import read_feature_dfs

from util.constants import LEARNING_SET, FULL_TEST_SET, ENTROPY_FEATURES
from util.visualization import plot_trainings_history, plot_rul_comparisons
from util.helper import *

from rul_prediction.ffnn import fit_ffnn


def entropy_features_no_classifier_ffnn():
    feature_list = ENTROPY_FEATURES
    learning_set = read_feature_dfs(data_set_sub_set=LEARNING_SET, features=feature_list)
    learning_set = concat_dfs(learning_set)
    training_labes = learning_set.pop('RUL')

    ffnn, trainings_history = fit_ffnn(X=learning_set, y=training_labes, epochs=100)

    plot_trainings_history(trainings_history=trainings_history, error_type='RMSE')

    testing_set = read_feature_dfs(data_set_sub_set=FULL_TEST_SET, features=feature_list)
    testing_labels = pop_labels(testing_set)

    plot_rul_comparisons(testing_set, testing_labels, ffnn)
