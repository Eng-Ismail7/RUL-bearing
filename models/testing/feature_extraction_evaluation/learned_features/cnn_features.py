import pandas as pd
import numpy as np

from pathlib import Path
from pre_processing.features import read_feature_dfs
from util.helper import pop_labels, reformat_flattened_data
from rul_prediction.cnn import fit_cnn
from util.constants import LEARNING_SET, FULL_TEST_SET, SPECTRA_CSV_NAME
from util.visualization import plot_rul_comparisons, plot_trainings_history
from tensorflow import keras


def spectra_features_no_classifier_cnn_rul_prediction(train: bool=True):
    model_path: str = Path('keras_models').joinpath('spectra_none_cnn')
    n_rows: int = 129
    n_cols: int = 21
    spectra_shape: tuple = (n_rows, n_cols)
    input_shape: tuple = (n_rows, n_cols, 1)

    if train:
        # Read in training data
        # print("Read in training data")
        read_spectra_dfs = read_feature_dfs(LEARNING_SET, SPECTRA_CSV_NAME)
        spectra_dfs = pd.concat(read_spectra_dfs, ignore_index=True, keys=['Bearing' + str(x)
                                                                           for x in range(0, len(read_spectra_dfs))])
        labels = spectra_dfs.pop('RUL')

        # Reformat flattened spectra
        spectra_dfs = spectra_dfs.to_numpy()
        spectra_dfs = np.array([df.reshape(spectra_shape) for df in spectra_dfs])

        # Train and save CNN
        # print("Train and save CNN")
        trainings_history, cnn = fit_cnn(spectra_dfs, labels, input_shape=input_shape, epochs=20)

        # Visualize training history
        plot_trainings_history(trainings_history)

        cnn.save(model_path)
    else:
        # Load pre-trained CNN model
        cnn = keras.models.load_model(model_path)

    # Visualize predicted RUL in comparison to real RUL of learning set
    comparison_set = read_feature_dfs(FULL_TEST_SET, SPECTRA_CSV_NAME)
    label_data = pop_labels(comparison_set)
    reshaped_comparison_set = reformat_flattened_data(comparison_set, n_rows=n_rows, n_cols=n_cols)
    plot_rul_comparisons(reshaped_comparison_set, label_data, cnn)


if __name__ == '__main__':
    spectra_features_no_classifier_cnn_rul_prediction(train=False)
