from pre_processing.features import read_feature_dfs
from util.constants import LEARNING_SET
from util.helper import concat_dfs, pop_labels
from rul_prediction.ffnn import fit_ffnn
from util.visualization import plot_trainings_history


def statistical_frequency_features_no_classifier_ffnn_rul_prediction():
    trainings_data = read_feature_dfs(LEARNING_SET)
    trainings_data = concat_dfs(trainings_data)
    trainings_labels = trainings_data.pop('RUL')
    filtered_cols: list = [column for column in trainings_data if column.startswith('frequency_')]
    trainings_data = trainings_data[filtered_cols]
    ffnn, trainings_history = fit_ffnn(trainings_data, trainings_labels, dropout=True, epochs=100, hidden_units=1024,
                                       hidden_layers=4)
    # Visualize trainings history
    plot_trainings_history(trainings_history)




if __name__ == '__main__':
    statistical_frequency_features_no_classifier_ffnn_rul_prediction()
