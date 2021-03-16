"""
File contains visualization methods.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path
from scipy.signal import savgol_filter
from tensorflow.keras.callbacks import History

from pre_processing.raw_features import read_raw_dfs_as_dict
from health_stage_classification.health_stage_classifiers import linear_rectification_technique
from util.constants import LEARNING_SET
from util.logging import store_dict
from util.metrics import standard_deviation


def plot_trainings_history(trainings_history: History, error_type: str = 'MAE'):
    """
    Plots the training history of a keras model.
    :param trainings_history: keras history object
    :param error_type: string that includes loss name
    :return: Void, shows plot
    """
    plt.plot(trainings_history.history['loss'], label=error_type + ' (training data)')

    plt.title(error_type + ' for RUL prediction')
    plt.ylabel(error_type + ' value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


def plot_keras_trainings_history(trainings_history: History, error_type: str = 'MAE', experiment_name: str = None,
                                 model_name: str = None):
    """
    Plots the training history of a keras model.
    :param trainings_history: keras history object
    :param error_type: string that includes loss name
    :param model_name if set stores the plot under 'pictures/pyplot/'+save_name+'.png'
    :return: Void, shows plot
    """
    plt.plot(trainings_history.history['loss'], label=error_type + ' (training data)')
    plt.plot(trainings_history.history['val_loss'], label=error_type + ' (validation data)')

    plt.title(error_type + ' for RUL prediction')
    plt.ylabel(error_type + ' value')
    plt.xlabel('No. epoch')
    plt.legend(['train', 'test'], loc="upper right")
    if model_name is None or experiment_name is None:
        plt.show()
    else:
        path_out = Path('pictures').joinpath('pyplot').joinpath('training_history').joinpath(experiment_name)
        store_dict(dict(trainings_history.history), experiment_name=experiment_name, kind_of_dict="training_history")
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath(model_name + '.png')
        plt.savefig(path_out, dpi=300)
        plt.clf()


def plot_rul_comparisons(bearing_data: Dict[str, pd.DataFrame], label_data: Dict[str, pd.Series],
                         prediction_model, use_svr: bool = False, use_gpr: bool = False,
                         use_poly_reg: bool = False, experiment_name: str = None, model_name: str = None):
    """
    Plot the real RUL in comparison to the RUL predicted by a Keras Model of multiple data frames.
    :param bearing_data: list of feature_dfs which RULs are to be predicted
    :param prediction_model: model used for prediction
    :return: Void, plots Facet grid which plots predicted and real RUL for each data frame
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12, 9))
    n: int = len(bearing_data)
    sqr: int = isqrt(n)
    if use_svr:
        prediction_dict = prediction_model.predict_svr(bearing_data)
    elif use_gpr:
        prediction_dict = prediction_model.predict_gpr(bearing_data)
    elif use_poly_reg:
        prediction_dict = prediction_model.predict_poly_reg(bearing_data)
    else:
        prediction_dict = prediction_model.predict(bearing_data)
    count = 1
    for key in prediction_dict.keys():
        predictions = prediction_dict[key]
        rul = label_data[key]
        # Smooth predictions
        predictions = savgol_filter(predictions, 9, 3)
        plt.subplot(sqr, sqr, count)
        sns.lineplot(data=rul)
        sns.lineplot(x=rul.index, y=predictions, size=0.1)
        plt.xlabel("Observation")
        plt.ylabel("RUL in Seconds")
        plt.legend([], [], frameon=False)
        plt.title(key.replace("_", " "))
        count += 1
    plt.tight_layout()
    if model_name is None or experiment_name is None:
        plt.show()
    else:
        path_out = Path('pictures').joinpath('pyplot').joinpath('rul_comparison').joinpath(experiment_name)
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath(model_name + '.png')
        plt.savefig(path_out, dpi=300)
        plt.clf()


# Helper
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    if x * x == n:
        return x
    else:
        return x + 1


def flatten_predictions(prediction_list: list) -> list:
    return [prediction[0] for prediction in prediction_list]


def plot_fpts(first_prediction_times: Dict[str, int], df_dict: Dict[str, pd.DataFrame], classification_indicator: str):
    """
    Plot the first prediction times on a specified feature of multiple data frames.
    :param first_prediction_times: list of first prediction times in order of the df_list
    :param df_list: list of feature data frames
    :param classification_indicator: features that is plotted, ideally the feature that is used to compute the FPT
    :return: Void, plots Facet grid which plots feature and FPT for each data frame
    """
    n: int = len(df_dict.keys())
    sqr: int = isqrt(n)

    count = 1
    for bearing, df in df_dict.items():
        indicator = df[classification_indicator]
        x_axis = indicator.index
        plt.subplot(sqr, sqr, count)
        sns.lineplot(x=x_axis, y=indicator)
        plt.xlabel(bearing.replace("_", " "))
        plt.axvline(x=first_prediction_times[bearing], color='red')
        count += 1
    plt.tight_layout()
    plt.show()


def plot_frequency_heatmap(zxx, f, t):
    plt.pcolormesh(t, f, np.abs(zxx), shading='gouraud')
    plt.title('Spectrum Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


def plot_raw_features():
    """
    Read all raw horizontal vibration observations, concatenate them for each bearing and plot
    :return:
    """
    raw_dfs_dict = read_raw_dfs_as_dict(sub_set=LEARNING_SET)
    _ = [df.pop('RUL') for _, df in raw_dfs_dict.items()]
    raw_dfs_list = [df.to_numpy().ravel() for _, df in raw_dfs_dict.items()]
    raw_dfs_list = [arr[0::2] for arr in raw_dfs_list]

    n: int = len(raw_dfs_list)
    sqr: int = isqrt(n)

    count = 1
    for i in range(len(raw_dfs_list)):
        df = raw_dfs_list[i]
        plt.subplot(sqr, sqr, count)
        sns.lineplot(data=df, size=0.1)
        plt.legend([], [], frameon=False)
        count += 1
    plt.tight_layout()
    plt.show()


def plot_metric_bar_overview(metric_data: Dict[str, Dict[str, Dict[str, float]]], metric_key: str,
                             experiment_name: str = None):
    # set width of bar
    bar_width = 10
    space_between_groups = bar_width * 2
    groups = list(metric_data.get(list(metric_data.keys())[0]).keys())  # get bearing keys

    # Set amount of members per group
    group_members = metric_data.keys()
    x_bar_left = np.array(
        [(bar_width * len(group_members) + space_between_groups) * i for i in
         range(len(groups))])

    offset = - len(group_members) / 2
    for member in group_members:
        y_values = [metric_dict.get(metric_key) for bearing, metric_dict in metric_data.get(member).items()]
        plt.bar(x_bar_left + offset * bar_width, y_values, width=bar_width, label=member.replace("_", " "),
                edgecolor='black')
        offset += 1

    plt.ylabel(metric_key)
    plt.xlabel("Bearings")
    plt.xticks(x_bar_left, [group[7:].replace("_", " ") for group in groups])
    plt.legend()

    if experiment_name is None:
        plt.savefig('test.png')
        plt.show()
    else:
        path_out = Path('pictures').joinpath('pyplot').joinpath('metrics').joinpath(experiment_name)
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath(metric_key + '.png')
        plt.savefig(path_out, dpi=300)
        plt.clf()


def plot_aggregated_metrics(metric_data: Dict[str, Dict[str, Dict[str, float]]],
                            experiment_name: str = None):
    bar_width = 0.5
    # plt.tight_layout()
    models = list(metric_data.keys())
    bearings = list(metric_data.get(models[0]).keys())
    metrics = list(metric_data.get(models[0]).get(bearings[0]).keys())
    x = np.arange(len(models))
    subplot_count = 1
    for metric_key in metrics:
        plt.subplot(1, len(metrics), subplot_count)
        count = 0
        for model in models:
            model_metrics = metric_data.get(model)
            metric_values_list = []
            for bearing in model_metrics.keys():
                metric_values_list += [model_metrics.get(bearing).get(metric_key)]
            std_dev = standard_deviation(metric_values_list)
            plt.bar(x[count], height=sum(metric_values_list) / len(metric_values_list), width=bar_width, yerr=std_dev)
            count += 1

        plt.ylabel(metric_key)
        plt.xlabel("Models")
        plt.xticks(x, [model.replace("_", " ") for model in models], fontsize=12)
        subplot_count += 1
    if experiment_name is None:
        plt.show()
    else:
        path_out = Path('pictures').joinpath('pyplot').joinpath('metrics').joinpath(experiment_name)
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath('aggregated.png')
        plt.savefig(path_out, dpi=300)
        plt.clf()


if __name__ == '__main__':
    plot_raw_features()
