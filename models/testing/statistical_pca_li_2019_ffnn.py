import pandas as pd

from pre_processing import read_feature_dfs
from util.helper import pop_labels, concat_dfs
from health_stage_classification.health_stage_classifiers import cut_fpts
from rul_prediction.ffnn import fit_ffnn
from util.constants import LEARNING_SET, FEATURES_CSV_NAME, FULL_TEST_SET, BASIC_STATISTICAL_FEATURES
from rul_features.learned_features.unsupervised.principal_component_analysis import pca_embedded_data_frame
from util.visualization import plot_rul_comparisons, plot_trainings_history


def statistical_features_and_pca_li_2019_classifier_ffnn_rul_prediction():
    # Input features: statistical features
    learning_feature_df_list = read_feature_dfs(LEARNING_SET, FEATURES_CSV_NAME, features=BASIC_STATISTICAL_FEATURES)

    # Two-Stage: lei et al 2019
    cut_dfs, first_prediction_times = cut_fpts(learning_feature_df_list)
    # Visualize FPTs
    # plot_fpts(first_prediction_times, learning_feature_df_list, 'root_mean_square')

    # Concatenate trainings data
    all_bearings = concat_dfs(cut_dfs)
    labels = all_bearings.pop('RUL')
    all_bearings, pca = pca_embedded_data_frame(all_bearings)

    # RUL prediction: FFNN
    trainings_history, ffnn = fit_ffnn(X=all_bearings, y=labels, dropout=True, epochs=150)

    # Visualize training history and later validation history
    plot_trainings_history(trainings_history)
    # Visualize predicted RUL in comparison to real RUL
    comparison_set = read_feature_dfs(FULL_TEST_SET, FEATURES_CSV_NAME, features=BASIC_STATISTICAL_FEATURES)
    comparison_set, first_prediction_times = cut_fpts(comparison_set)
    # Remove label
    label_data = pop_labels(comparison_set)
    # Apply PCA
    comparison_set = [pd.DataFrame(pca.transform(df)) for df in comparison_set]

    plot_rul_comparisons(comparison_set, label_data=label_data, prediction_model=ffnn)
