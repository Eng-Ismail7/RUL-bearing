import pandas as pd
from tqdm import tqdm
from rul_prediction.ffnn import fit_ffnn
from pre_processing.raw_features import read_raw_dfs
from util.constants import LEARNING_SET, FULL_TEST_SET
from util.helper import concat_dfs, pop_labels
from util.visualization import plot_rul_comparisons, plot_trainings_history
from rul_features.learned_features.unsupervised.principal_component_analysis import pca_embedded_data_frame


def pca_features_no_classifier_ffnn_rul_prediction(training_data, comparison_set):
    hi = __name__
    # Read Training data
    # Remove labels
    pca_training_data = concat_dfs(training_data)
    labels = pca_training_data.pop('RUL')
    model_training_data, pca = pca_embedded_data_frame(pca_training_data, n_components=300, verbose=False)

    model_training_data = pd.DataFrame(model_training_data)
    ffnn, training_history = fit_ffnn(model_training_data, labels, epochs=60, dropout=True)

    plot_trainings_history(training_history)

    # Visualize predicted RUL in comparison to real RUL of training set
    # Remove label
    training_label_data = pop_labels(training_data)
    # Apply PCA
    transformed_training_set = []
    for df in tqdm(training_data, desc="Transforming training set.", position=0, leave=True):
        transformed_training_set += [pd.DataFrame(pca.transform(X=df))]
    plot_rul_comparisons(transformed_training_set, label_data=training_label_data, prediction_model=ffnn)

    # Visualize predicted RUL in comparison to real RUL of validation set
    # Remove label
    label_data = pop_labels(comparison_set)
    # Apply PCA
    transformed_comparison_set = []
    for df in tqdm(comparison_set, desc="Transforming validation set.", position=0, leave=True):
        transformed_comparison_set += [pd.DataFrame(pca.transform(X=df))]

    plot_rul_comparisons(transformed_comparison_set, label_data=label_data, prediction_model=ffnn)


if __name__ == '__main__':
    pca_features_no_classifier_ffnn_rul_prediction()
