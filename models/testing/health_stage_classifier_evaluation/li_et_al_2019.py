from pre_processing.features import read_feature_dfs_as_dict

from util.constants import LEARNING_SET, FULL_TEST_SET
from util.visualization import plot_trainings_history, plot_rul_comparisons, plot_fpts
from util.helper import *

from rul_prediction.ffnn import fit_ffnn
from health_stage_classification.health_stage_classifiers import li_et_al_2019, cut_fpts


def tba_features_li_et_al_2019_ffnn():
    feature_list = []

    fpt_method = li_et_al_2019
    signal_key = "h_kurtosis"
    learning_set = read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET)

    # Calculate FPTs and cut dataframes
    cut_learning_set, first_prediction_times = cut_fpts(learning_set, fpt_method=fpt_method,
                                                        signal_key=signal_key)

    plot_fpts(first_prediction_times=first_prediction_times, df_dict=learning_set,
              classification_indicator="h_kurtosis")

if __name__ == '__main__':
    tba_features_li_et_al_2019_ffnn()