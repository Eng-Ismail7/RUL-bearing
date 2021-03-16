import pandas as pd

from pre_processing import read_dfs
from health_stage_classification.health_stage_classifiers import cut_fpts, procentual_rul
from rul_prediction.ffnn import fit_ffnn
from util.constants import LEARNING_SET, ALL_FEATURES, FULL_TEST_SET

from util.visualization import plot_rul_comparisons, plot_trainings_history, plot_fpts


def all_features_li_2019_classifier_ffnn_rul_prediction():
    training_set = read_dfs(LEARNING_SET, ALL_FEATURES)
    # Two-Stage: lei et al 2019
    training_set, first_prediction_times = cut_fpts(training_set)
    training_set = procentual_rul(training_set, first_prediction_times)

    # Visualize FPTs
    plot_fpts(first_prediction_times, training_set, 'fourier_entropy')

    # Concatenate trainings data
    concatenated_training_set = pd.concat(training_set, ignore_index=True, keys=['Bearing' + str(x)
                                                                                 for x in range(0, len(training_set))])
    labels = concatenated_training_set.pop('RUL')

    # Train FFNN
    trainings_history, ffnn = fit_ffnn(X=concatenated_training_set, y=labels, dropout=True, epochs=50)

    # Visualize training history and later validation history
    plot_trainings_history(trainings_history)
    # Visualize predicted RUL in comparison to real RUL of learning set
    plot_rul_comparisons(training_set, prediction_model=ffnn)

    # TESTING ESTIMATIONS #
    testing_set = read_dfs(FULL_TEST_SET, ALL_FEATURES)
    testing_set, fpts = cut_fpts(testing_set)
    testing_set = procentual_rul(testing_set, fpts)
    plot_rul_comparisons(testing_set, prediction_model=ffnn)


if __name__ == '__main__':
    all_features_li_2019_classifier_ffnn_rul_prediction()
