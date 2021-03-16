from pre_processing.features import read_feature_dfs
from util.helper import concat_dfs, pop_labels
from util.visualization import plot_trainings_history
from util.constants import LEARNING_SET
from rul_prediction.lstm import fit_lstm


def all_features_no_classifier_lstm_rul_prediction():
    trainings_data = read_feature_dfs(LEARNING_SET)
    #trainings_data = concat_dfs(trainings_data)
    trainings_labels = pop_labels(trainings_data)

    lstm, trainings_history = fit_lstm(trainings_data, trainings_labels, dropout=True)
    plot_trainings_history(trainings_history=trainings_history)


if __name__ == '__main__':
    all_features_no_classifier_lstm_rul_prediction()
