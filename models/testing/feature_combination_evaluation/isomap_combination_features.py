import pandas as pd
from tqdm import tqdm

from rul_prediction.ffnn import fit_ffnn
from pre_processing.features import read_feature_dfs
from util.constants import LEARNING_SET, FULL_TEST_SET, ALL_FEATURES
from util.helper import concat_dfs, pop_labels
from util.visualization import plot_rul_comparisons, plot_trainings_history
from rul_features.learned_features.unsupervised.isomap import isomap_embedded_data_frame


def all_features_isomap_combination_no_classifier_ffnn_rul_prediction():
    # Read Training data
    feature_list = ALL_FEATURES
    training_data = read_feature_dfs(data_set_sub_set=LEARNING_SET, features=feature_list)
    learning_set = concat_dfs(training_data)
    training_labes = learning_set.pop('RUL')

    # Remove labels
    model_training_data, isomap = isomap_embedded_data_frame(learning_set, verbose=False)

    model_training_data = pd.DataFrame(model_training_data)
    ffnn, training_history = fit_ffnn(model_training_data, training_labes, epochs=60)  # TODO dropout=True

    plot_trainings_history(training_history)

    # Visualize predicted RUL in comparison to real RUL of training set
    # Remove label
    training_label_data = pop_labels(training_data)
    # Apply PCA
    transformed_training_set = []
    for df in tqdm(training_data, desc="Transforming validation set.", position=0, leave=True):
        transformed_training_set += [pd.DataFrame(isomap.transform(X=df))]
    plot_rul_comparisons(transformed_training_set, label_data=training_label_data, prediction_model=ffnn)

    # Visualize predicted RUL in comparison to real RUL of validation set
    comparison_set = read_feature_dfs(data_set_sub_set=FULL_TEST_SET, features=feature_list)
    # Remove label
    label_data = pop_labels(comparison_set)
    # Apply PCA
    transformed_comparison_set = []
    for df in tqdm(comparison_set, desc="Transforming validation set.", position=0, leave=True):
        transformed_comparison_set += [pd.DataFrame(isomap.transform(X=df))]

    plot_rul_comparisons(transformed_comparison_set, label_data=label_data, prediction_model=ffnn)


if __name__ == '__main__':
    isomap_features_no_classifier_ffnn_rul_prediction()
