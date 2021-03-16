import os
import json
from pathlib import Path
from typing import Dict, List, Iterable
from util.constants import RMSE_KEY, CORR_COEFF_KEY, METRICS_DICT_PATH
from util.metrics import standard_deviation

table_end = "\end{tabular}"
row_end = "\\\\"
line_break = "\n"
ampersand = " & "


def save_latex_aggregated_table(metric_data, experiment_name: str = None):
    models = list(metric_data.keys())
    bearings = list(metric_data.get(models[0]).keys())
    metrics = list(metric_data.get(models[0]).get(bearings[0]).keys())
    model_to_metric_dict = {}
    for model in models:
        metric_dict = {}
        for metric in metrics:
            metric_values_list = []
            average = 0
            std_dev = 0
            for bearing in bearings:
                metric_values_list += [metric_data.get(model).get(bearing).get(metric)]
                average = sum(metric_values_list) / len(metric_values_list)
                std_dev = standard_deviation(metric_values_list)
            metric_dict[metric] = average
            metric_dict[metric+"_std"] = std_dev
        model_to_metric_dict[model] = metric_dict

    latex_string = begin_latex_table(headers=["Metric"] + models)
    for metric in metrics:
        latex_string += metric
        for model in model_to_metric_dict:
            latex_string += ampersand + "{:14.2f}".format(model_to_metric_dict.get(model).get(metric))
            latex_string += " $\pm$"+"{:14.2f}".format(model_to_metric_dict.get(model).get(metric+"_std"))
        latex_string += row_end + line_break
    latex_string += table_end
    latex_string = latex_string.replace("_", " ")

    if experiment_name is None:
        print(latex_string)
    else:
        path_out = Path('logs').joinpath('aggregated_metrics_table')
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        with open(path_out.joinpath(experiment_name + '.txt'), 'w') as out_file:
            out_file.write(latex_string)


def save_latex_grouped_metrics_table(metrics_dict: Dict[str, Dict[str, Dict[str, float]]], experiment_name: str = None):
    model_level_keys: List[str] = list(metrics_dict.keys())
    bearing_level_keys: List[str] = list(metrics_dict.get(model_level_keys[0]))
    metrics_level_keys: List[str] = list(metrics_dict.get(model_level_keys[0]).get(bearing_level_keys[0]))

    headers = ["Bearing", "Metric"] + model_level_keys
    latex_string = begin_latex_table(headers)

    for bearing in bearing_level_keys:
        models_then_metrics_list: Dict[str, Dict[str, str]] = \
            {model:
                {metric: "{:14.2f}".format(
                    metrics_dict.get(model).get(bearing).get(metric))
                    for metric in metrics_level_keys} for
                model in model_level_keys}
        latex_multirow = "\multirow{" + str(len(metrics_level_keys)) + "}{*}{" + bearing + "}"
        latex_multirow += add_rows_to_multirow(metrics_level_keys, models_then_metrics_list=models_then_metrics_list)
        latex_string += line_break + latex_multirow + "\hline" + line_break

    latex_string += table_end

    latex_string = latex_string.replace("_", " ")
    latex_string = latex_string.replace("Bearing", "")
    if experiment_name is None:
        print(latex_string)
    else:
        path_out = Path('logs').joinpath('grouped_metrics_table')
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        with open(path_out.joinpath(experiment_name + ".txt"), "w") as text_file:
            text_file.write(latex_string)


def add_rows_to_multirow(metrics_keys: List[str], models_then_metrics_list: Dict[str, Dict[str, str]]) -> str:
    result = ""
    for metric in metrics_keys:
        result += ampersand + metric
        wide = ""
        underline = ""
        values = sorted([float(models_then_metrics_list.get(model).get(metric)) for model in models_then_metrics_list])
        if metric == RMSE_KEY:
            wide = "{:14.2f}".format(values[0])
            underline = "{:14.2f}".format(values[1])

        if metric == CORR_COEFF_KEY:
            wide = "{:14.2f}".format(values[-1])
            underline = "{:14.2f}".format(values[-2])

        for model in models_then_metrics_list.keys():
            curr_value = models_then_metrics_list.get(model).get(metric)
            if curr_value == wide:
                curr_value = "\\textbf{" + curr_value + "}"
            if curr_value == underline:
                curr_value = "\\underline{" + curr_value + "}"
            result += ampersand + curr_value
        result += row_end + line_break
    return result


def begin_latex_table(headers: List[str]):
    table_start = "\\begin{tabular}{" + "|".join(
        ["c" for _ in range(len(headers))]) + "}" + line_break + "\hline" + line_break
    table_start += ampersand.join(headers) + row_end + line_break + "\hline" + line_break
    return table_start


def store_metrics_dict(dict: Dict[str, Dict[str, Dict[str, float]]], experiment_name: str):
    path_out = Path(METRICS_DICT_PATH)
    if not os.path.exists(path_out):
        Path(path_out).mkdir(parents=True, exist_ok=True)
    with open(path_out.joinpath(experiment_name + ".json"), "w") as text_file:
        json.dump(dict, text_file, indent=4)


def store_dict(dict: Dict[str, Dict[str, Dict[str, float]]], experiment_name: str, kind_of_dict: str):
    path_out = Path('logs').joinpath(kind_of_dict)
    if not os.path.exists(path_out):
        Path(path_out).mkdir(parents=True, exist_ok=True)
    with open(path_out.joinpath(experiment_name + ".json"), "w") as text_file:
        json.dump(dict, text_file, indent=4)


if __name__ == '__main__':
    metrics_dict = {
        'statistical': {
            'Bearing1_3': {'Root Mean Square Error': 57535.98828862467,
                           'Correlation Coefficient': -0.22474967285861494},
            'Bearing1_4': {'Root Mean Square Error': 182548.2555591824, 'Correlation Coefficient': -0.6668890345725328},
            'Bearing1_5': {'Root Mean Square Error': 13258.93375250344,
                           'Correlation Coefficient': -0.10760729111474987},
            'Bearing1_6': {'Root Mean Square Error': 13257.62385749784,
                           'Correlation Coefficient': -0.16753528048013383},
            'Bearing1_7': {'Root Mean Square Error': 12136.813263243768,
                           'Correlation Coefficient': -0.3488072413247197},
            'Bearing2_3': {'Root Mean Square Error': 10734.295522598877,
                           'Correlation Coefficient': -0.05958293360573337},
            'Bearing2_4': {'Root Mean Square Error': 3846.717163441271,
                           'Correlation Coefficient': -0.22886090036784532},
            'Bearing2_5': {'Root Mean Square Error': 12600.446042911197,
                           'Correlation Coefficient': -0.23079787488731138},
            'Bearing2_6': {'Root Mean Square Error': 3788.988129304716,
                           'Correlation Coefficient': -0.21824074124916082},
            'Bearing2_7': {'Root Mean Square Error': 2054.844860567375,
                           'Correlation Coefficient': -0.28457201153956035},
            'Bearing3_3': {'Root Mean Square Error': 3147.697551193578,
                           'Correlation Coefficient': -0.6494021429092784}},
        'entropy': {
            'Bearing1_3': {'Root Mean Square Error': 7277.923760799836,
                           'Correlation Coefficient': 0.7839526476882971},
            'Bearing1_4': {'Root Mean Square Error': 4491.451722386021,
                           'Correlation Coefficient': -0.09372609666294451},
            'Bearing1_5': {'Root Mean Square Error': 7852.019027180953,
                           'Correlation Coefficient': -0.31467535345046094},
            'Bearing1_6': {'Root Mean Square Error': 7848.812508887416,
                           'Correlation Coefficient': -0.41540843719666065},
            'Bearing1_7': {'Root Mean Square Error': 6922.795747844454,
                           'Correlation Coefficient': 0.23880968137184283},
            'Bearing2_3': {'Root Mean Square Error': 5834.34026755728,
                           'Correlation Coefficient': -0.3435631712839462},
            'Bearing2_4': {'Root Mean Square Error': 5433.134243272633,
                           'Correlation Coefficient': -0.5994781345901937},
            'Bearing2_5': {'Root Mean Square Error': 7320.517696689965,
                           'Correlation Coefficient': -0.6299458133765713},
            'Bearing2_6': {'Root Mean Square Error': 5800.348227499665,
                           'Correlation Coefficient': -0.7185918785522664},
            'Bearing2_7': {'Root Mean Square Error': 7700.502773322415,
                           'Correlation Coefficient': 0.10791643816838813},
            'Bearing3_3': {'Root Mean Square Error': 6729.672171470464,
                           'Correlation Coefficient': 0.4756485239778823}},
        'frequency': {
            'Bearing1_3': {'Root Mean Square Error': 28481.447080809834,
                           'Correlation Coefficient': -0.3711903178268098},
            'Bearing1_4': {'Root Mean Square Error': 4481.334902369774, 'Correlation Coefficient': 0.1165545404056595},
            'Bearing1_5': {'Root Mean Square Error': 10566.816721029, 'Correlation Coefficient': -0.4673174846852846},
            'Bearing1_6': {'Root Mean Square Error': 9834.045157240298,
                           'Correlation Coefficient': -0.33129323477275424},
            'Bearing1_7': {'Root Mean Square Error': 9665.06783945159, 'Correlation Coefficient': 0.015659790404419154},
            'Bearing2_3': {'Root Mean Square Error': 8713.850607500968,
                           'Correlation Coefficient': -0.14968946406296976},
            'Bearing2_4': {'Root Mean Square Error': 7096.388761914862, 'Correlation Coefficient': -0.0871144120167853},
            'Bearing2_5': {'Root Mean Square Error': 8478.583135970728, 'Correlation Coefficient': 0.26898036846397855},
            'Bearing2_6': {'Root Mean Square Error': 5492.800931093873,
                           'Correlation Coefficient': -0.026132119296600864},
            'Bearing2_7': {'Root Mean Square Error': 9204.15543384288, 'Correlation Coefficient': -0.5321828589268536},
            'Bearing3_3': {'Root Mean Square Error': 11221.510888635581,
                           'Correlation Coefficient': -0.011160949175702489}}, 'all': {
            'Bearing1_3': {'Root Mean Square Error': 36806.88349776104,
                           'Correlation Coefficient': -0.16975470805119966},
            'Bearing1_4': {'Root Mean Square Error': 79419.20785034969, 'Correlation Coefficient': -0.6710538139776491},
            'Bearing1_5': {'Root Mean Square Error': 13029.250603123568,
                           'Correlation Coefficient': -0.4255237479896216},
            'Bearing1_6': {'Root Mean Square Error': 13033.117205531591,
                           'Correlation Coefficient': -0.36067222987122305},
            'Bearing1_7': {'Root Mean Square Error': 11936.631449951803,
                           'Correlation Coefficient': -0.35858092215437865},
            'Bearing2_3': {'Root Mean Square Error': 10013.342380874881,
                           'Correlation Coefficient': -0.12046537556193841},
            'Bearing2_4': {'Root Mean Square Error': 3387.784861699398,
                           'Correlation Coefficient': -0.27089296534773755},
            'Bearing2_5': {'Root Mean Square Error': 12367.495314497704,
                           'Correlation Coefficient': -0.4383831863676729},
            'Bearing2_6': {'Root Mean Square Error': 3113.337087536866, 'Correlation Coefficient': -0.2505594044561476},
            'Bearing2_7': {'Root Mean Square Error': 1215.5251801627803,
                           'Correlation Coefficient': -0.4453961486354302},
            'Bearing3_3': {'Root Mean Square Error': 2057.5302803229424,
                           'Correlation Coefficient': -0.5511725922890299}}, 'CNN': {
            'Bearing1_3': {'Root Mean Square Error': 11476.06679304357, 'Correlation Coefficient': 0.5754882998127293},
            'Bearing1_4': {'Root Mean Square Error': 6501.235354446564,
                           'Correlation Coefficient': -0.45098137906857033},
            'Bearing1_5': {'Root Mean Square Error': 11981.514850726406, 'Correlation Coefficient': 0.5752891518450516},
            'Bearing1_6': {'Root Mean Square Error': 12155.442327886407, 'Correlation Coefficient': 0.6513588393932912},
            'Bearing1_7': {'Root Mean Square Error': 10838.898892024392,
                           'Correlation Coefficient': 0.42635210361837067},
            'Bearing2_3': {'Root Mean Square Error': 9349.638772164102, 'Correlation Coefficient': 0.48069773370481717},
            'Bearing2_4': {'Root Mean Square Error': 2540.380511173322, 'Correlation Coefficient': 0.6901504814146379},
            'Bearing2_5': {'Root Mean Square Error': 11457.369271121697,
                           'Correlation Coefficient': 0.47368323461461437},
            'Bearing2_6': {'Root Mean Square Error': 2368.5795837107976,
                           'Correlation Coefficient': 0.03829367068038065},
            'Bearing2_7': {'Root Mean Square Error': 1591.3476003594903, 'Correlation Coefficient': 0.675707929369903},
            'Bearing3_3': {'Root Mean Square Error': 1161.5361012810301,
                           'Correlation Coefficient': 0.4575306842882795}}}
    # save_latex_grouped_metrics_table(metrics_dict)
    save_latex_aggregated_table(metrics_dict, "test_experiment")
