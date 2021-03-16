from typing import Sequence, Callable, Dict
import pandas as pd
import abc
from models.DataSetType import DataSetType

from util.visualization import plot_rul_comparisons
from util.helper import pop_labels, flatten_predictions
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History

from rul_prediction.ffnn import fit_ffnn


class DegradationModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name: str, data_set_type: DataSetType):
        self.prediction_model: Model = None
        self.svr = None
        self.gpr = None
        self.poly_reg = None
        self.trainings_history = None
        self.name = name
        self.data_set_type = data_set_type

    @abc.abstractmethod
    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Takes a list of input vectors and returns RUL predictions for that list.
        :param df_dict:
        :return:
        """
        pass

    @abc.abstractmethod
    def predict_svr(self, df_dict: Dict[str, pd.DataFrame]):
        pass
    
    @abc.abstractmethod
    def predict_gpr(self, df_dict: Dict[str, pd.DataFrame]):
        pass

    @abc.abstractmethod
    def predict_poly_reg(self, df_dict: Dict[str, pd.DataFrame]):
        pass

    @abc.abstractmethod
    def train(self, training_data: pd.DataFrame, training_labels: pd.Series, validation_data: pd.DataFrame,
              validation_labels: pd.Series) -> History:
        """
        Takes training data and labels and trains a model corresponding to the respective class.
        Also sets the prediction model.
        :return: Returns the trainings history
        """
        pass

    @abc.abstractmethod
    def train_svr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        pass

    def compute_metrics(self, df_dict: Dict[str, pd.DataFrame], labels: Dict[str, pd.Series],
                        metrics_list: Sequence[Callable], use_svr: bool = False, use_gpr: bool = False,
                        use_poly_reg: bool = False) -> Dict[str, Dict[str, float]]:
        if use_svr:
            predictions = self.predict_svr(df_dict=df_dict)
        elif use_gpr:
            predictions = self.predict_gpr(df_dict=df_dict)
        elif use_poly_reg:
            predictions = self.predict_poly_reg(df_dict=df_dict)
        else:
            predictions = self.predict(df_dict=df_dict)
        result = {}
        for key in df_dict.keys():
            result[key] = {metric.__name__: metric(labels[key], predictions[key]) for metric in metrics_list}
        return result

    def visualize_rul(self, df_dict: Dict[str, pd.DataFrame], label_data: Dict[str, pd.Series], experiment_name: str,
                      use_svr: bool = False, use_gpr: bool = False, use_poly_reg: bool = False):
        plot_rul_comparisons(df_dict, label_data, self, experiment_name=experiment_name, model_name=self.name,
                             use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    def get_name(self) -> str:
        return self.name

    def get_data_set_type(self) -> DataSetType:
        return self.data_set_type
