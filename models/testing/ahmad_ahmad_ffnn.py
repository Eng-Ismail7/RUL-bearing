from pre_processing.features import read_feature_dfs
from rul_prediction.ffnn import fit_ffnn
from health_stage_classification.health_stage_classifiers import cut_fpts, ahmad_et_al_2019
from util.constants import LEARNING_SET, FULL_TEST_SET
from util.helper import concat_dfs, pop_labels
from util.visualization import plot_fpts, plot_rul_comparisons
import pandas as pd


def ahmad_health_indicator_ahmad_classifier_ffnn_rul_prediction():
    training_set = read_feature_dfs(LEARNING_SET)
    training_labels = pop_labels(training_set)
    training_set = construct_ahmad_health_indicator(training_set)
    training_set = [pd.merge(training_set[i], training_labels[i], left_index=True, right_index=True) for i in
                    range(len(training_labels))]
    cut_dfs, fpts = cut_fpts(training_set, fpt_method=ahmad_et_al_2019, signal_key='ahmad_health_indicator')
    plot_fpts(fpts, df_list=training_set, classification_indicator='ahmad_health_indicator')

    training_set = concat_dfs(training_set)
    training_labels = training_set.pop('RUL')
    ffnn, trainin_history = fit_ffnn(training_set, training_labels, dropout=True, epochs=30, hidden_layers=3,
                                     hidden_units=128)

    comparison_set = read_feature_dfs(FULL_TEST_SET)
    comparison_set, first_prediction_times = cut_fpts(comparison_set)

    # Remove label
    label_data = pop_labels(comparison_set)
    plot_rul_comparisons(comparison_set, label_data=label_data, prediction_model=ffnn)


def construct_ahmad_health_indicator(df_list: list) -> list:
    rms_normal_list = [df.iloc[0:100]['root_mean_square'].mean() for df in df_list]
    result = []
    for i in range(len(rms_normal_list)):
        result += [pd.Series(df_list[i]['root_mean_square'] / rms_normal_list[i], name='ahmad_health_indicator')]
    return result


if __name__ == '__main__':
    ahmad_health_indicator_ahmad_classifier_ffnn_rul_prediction()
