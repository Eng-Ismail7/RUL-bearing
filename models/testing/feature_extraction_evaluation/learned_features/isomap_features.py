import pandas as pd

from rul_prediction.ffnn import fit_ffnn
from pre_processing.raw_features import read_raw_dfs
from util.constants import LEARNING_SET, FULL_TEST_SET
from util.helper import concat_dfs, pop_labels
from util.visualization import plot_rul_comparisons, plot_trainings_history
from rul_features.learned_features.unsupervised.isomap import isomap_embedded_data_frame


def isomap_features_no_classifier_ffnn_rul_prediction(training_data, comparison_set):
    # Read Training data
    isomap_training_data = training_data

    # Remove labels
    isomap_training_data = concat_dfs(isomap_training_data)
    labels = isomap_training_data.pop('RUL')
    model_training_data, isomap = isomap_embedded_data_frame(isomap_training_data, verbose=False)

    ffnn, training_history = fit_ffnn(model_training_data, labels, epochs=100)

    plot_trainings_history(training_history)
    # Visualize predicted RUL in comparison to real RUL
    #   comparison_set = read_raw_dfs(FULL_TEST_SET)
    # Remove label
    label_data = pop_labels(comparison_set)
    # Apply autoencoder
    comparison_set = [pd.DataFrame(isomap.transform(X=df)) for df in comparison_set]

    plot_rul_comparisons(comparison_set, label_data=label_data, prediction_model=ffnn)


if __name__ == '__main__':
    isomap_features_no_classifier_ffnn_rul_prediction()
