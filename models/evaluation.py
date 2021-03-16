import os
import copy
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Sequence, Dict

from pre_processing.features import read_feature_dfs_as_dict, df_dict_to_df_dataframe
from pre_processing.raw_features import read_raw_dfs_as_dict

from util.logging import save_latex_grouped_metrics_table, store_metrics_dict, save_latex_aggregated_table
from util.visualization import plot_keras_trainings_history, plot_metric_bar_overview, plot_aggregated_metrics
from util.helper import pop_labels
from util.metrics import rmse, correlation_coefficient

from util.constants import METRICS_DICT_PATH
from util.constants import LEARNING_SET, FULL_TEST_SET
from util.constants import SPECTRA_CSV_NAME
from util.constants import RMSE_KEY, CORR_COEFF_KEY
from util.constants import MEMORY_CACHE_PATH

from models.DegradationModel import DegradationModel
from models.DataSetType import DataSetType
from models.HealthStageClassifier import HealthStageClassifier


def create_plots_and_latex(experiment_name: str, health_stage_classifier: HealthStageClassifier = None,
                           use_svr: bool = False, use_gpr: bool = False, use_poly_reg: bool = False):
    if health_stage_classifier is not None:
        experiment_name += "_true"
    else:
        experiment_name += "_false"
    if use_svr:
        experiment_name += "_SVR"
    elif use_gpr:
        experiment_name += "_GPR"
    elif use_poly_reg:
        experiment_name += "_MLR"
    else:
        experiment_name += "_ANN"

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(16, 9))
    plt.rc("errorbar", capsize=5)
    plt.tight_layout()

    path_out = Path(METRICS_DICT_PATH)
    if not os.path.exists(path_out):
        Path(path_out).mkdir(parents=True, exist_ok=True)
    with open(path_out.joinpath(experiment_name + '.json'), 'r') as file:
        metric_data = json.load(file)
        plot_metric_bar_overview(metric_data=metric_data, metric_key=RMSE_KEY, experiment_name=experiment_name)
        plot_metric_bar_overview(metric_data=metric_data, metric_key=CORR_COEFF_KEY,
                                 experiment_name=experiment_name)

        plot_aggregated_metrics(metric_data=metric_data, experiment_name=experiment_name)


def do_eval(model_dict: Dict[str, Sequence[DegradationModel]], health_stage_classifier: HealthStageClassifier = None,
            use_svr: bool = False, use_gpr: bool = False, use_poly_reg: bool = False):
    assert not (use_svr and use_gpr and use_poly_reg)
    # Read evaluation data
    raw_metric_data = read_raw_dfs_as_dict(FULL_TEST_SET)
    feature_metric_data = read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET)
    spectra_metric_data = read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET, csv_name=SPECTRA_CSV_NAME)

    # Read Raw Data
    raw_training_data, raw_training_labels = df_dict_to_df_dataframe(read_raw_dfs_as_dict(LEARNING_SET))
    raw_validation_data, raw_validation_labes = df_dict_to_df_dataframe(copy.deepcopy(raw_metric_data))

    # Read Computed Feature Data
    feature_training_data, feature_training_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET))
    feature_validation_data, feature_validation_labels = df_dict_to_df_dataframe(copy.deepcopy(feature_metric_data))

    # Read Frequency Spectra Data
    spectra_training_dict, spectra_training_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET, csv_name=SPECTRA_CSV_NAME))
    spectra_validation_dict, spectra_validation_labels = df_dict_to_df_dataframe(copy.deepcopy(spectra_metric_data))

    training_data_dict: Dict[DataSetType, Sequence[pd.DataFrame]] = {
        DataSetType.raw: ((raw_training_data,
                           raw_training_labels),
                          (raw_validation_data,
                           raw_validation_labes)),
        DataSetType.computed: ((feature_training_data,
                                feature_training_labels),
                               (feature_validation_data,
                                feature_validation_labels)),
        DataSetType.spectra: ((spectra_training_dict,
                               spectra_training_labels),
                              (spectra_validation_dict,
                               spectra_validation_labels))}

    # Format validation data
    raw_metric_labels = pop_labels(raw_metric_data)
    feature_metric_labels = pop_labels(feature_metric_data)
    spectra_metric_labels = pop_labels(spectra_metric_data)

    validation_metric_data: Dict[DataSetType, Sequence[Dict[str, pd.DataFrame], Dict[str, pd.Series]]] = {
        DataSetType.raw: (raw_metric_data, raw_metric_labels),
        DataSetType.computed: (feature_metric_data, feature_metric_labels),
        DataSetType.spectra: (spectra_metric_data, spectra_metric_labels)
    }

    # Cut dfs according to health_stage_classifier
    if health_stage_classifier is not None:
        for key in training_data_dict.keys():
            training_data_frames = training_data_dict.get(key)
            new_datasets = []
            (training_data, training_labels), (validation_data, validation_labels) = training_data_frames
            new_datasets += [
                (health_stage_classifier.cut_FPTs_of_dataframe(training_data, training_labels, feature_training_data))]
            new_datasets += [(health_stage_classifier.cut_FPTs_of_dataframe(validation_data, validation_labels,
                                                                            feature_validation_data))]
            training_data_dict[key] = new_datasets

        fpt_dict = {}
        for key, (data, labels) in validation_metric_data.items():
            cut_data, cut_labels, fpts = health_stage_classifier.cut_FPTs_of_dataframe_dict(data, labels,
                                                                                            feature_validation_data)
            validation_metric_data[key] = (cut_data, cut_labels)
            fpt_dict[str(key)] = fpts
        fpt_path = Path("logs").joinpath("first_prediction_times")
        if not os.path.exists(fpt_path):
            Path(fpt_path).mkdir(parents=True, exist_ok=True)
        with open(fpt_path.joinpath(health_stage_classifier.name), 'w') as file:
            json.dump(fpt_dict, file, indent=4)

    # Evaluate Models
    for model_group in tqdm(model_dict.keys(), desc="Evaluating model groups"):
        experiment_name = model_group
        if health_stage_classifier is not None:
            experiment_name += "_true"
        else:
            experiment_name += "_false"
        if use_svr:
            experiment_name += "_SVR"
        elif use_gpr:
            experiment_name += "_GPR"
        elif use_poly_reg:
            experiment_name += "_MLR"
        else:
            experiment_name += "_ANN"
        model_list = model_dict.get(model_group)
        # Train Models
        for model in tqdm(model_list, desc="Training models for model group %s" % experiment_name):
            (training_data, training_labels), (validation_data, validation_labels) = training_data_dict.get(
                model.get_data_set_type())
            if use_svr:
                model.train_svr(training_data=training_data, training_labels=training_labels)
            elif use_gpr:
                model.train_gpr(training_data=training_data, training_labels=training_labels)
            elif use_poly_reg:
                model.train_poly_reg(training_data=training_data, training_labels=training_labels, memory_path=MEMORY_CACHE_PATH)
            else:
                trainings_history = model.train(training_data=training_data, training_labels=training_labels,
                                                validation_data=validation_data, validation_labels=validation_labels)

        metric_data = {}
        # Evaluate Models
        for model in tqdm(model_list, desc="Evaluating models for model group %s" % experiment_name, position=0):
            model_metric_data, model_metric_labels = validation_metric_data.get(model.get_data_set_type())
            metric_data[model.get_name()] = model.compute_metrics(df_dict=model_metric_data,
                                                                  labels=model_metric_labels,
                                                                  metrics_list=[rmse, correlation_coefficient],
                                                                  use_svr=use_svr,
                                                                  use_gpr=use_gpr,
                                                                  use_poly_reg=use_poly_reg)


            model.visualize_rul(model_metric_data, model_metric_labels, experiment_name=experiment_name,
                                use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)
        store_metrics_dict(dict=metric_data, experiment_name=experiment_name)
