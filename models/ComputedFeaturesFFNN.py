from typing import Dict, Sequence, Callable
import pandas as pd
from tensorflow.keras.callbacks import History

from util.helper import flatten_predictions
from rul_prediction.ffnn import fit_ffnn
from rul_prediction.suport_vector_regression import fit_svr
from rul_prediction.gpr import fit_gpr
from rul_prediction.poly_reg import fit_poly_reg
from models.DegradationModel import DegradationModel
from models.DataSetType import DataSetType

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class ComputedFeaturesFFNN(DegradationModel):
    def __init__(self, feature_list: Sequence[Callable], name: str):
        DegradationModel.__init__(self, name, DataSetType.computed)
        self.feature_list = [prefix + f.__name__ for f in feature_list for prefix in ['h_', 'v_']]

    def train(self, training_data: pd.DataFrame, training_labels: pd.Series, validation_data: pd.DataFrame,
              validation_labels: pd.Series) -> History:
        training_data = training_data[self.feature_list]
        validation_data = validation_data[self.feature_list]
        self.prediction_model, self.trainings_history = fit_ffnn(X=training_data, y=training_labels,
                                                                 dropout=True,
                                                                 validation_data=(validation_data, validation_labels))
        return self.trainings_history

    def train_svr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        training_data = training_data[self.feature_list]
        self.svr = fit_svr(training_data, training_labels)

    def train_gpr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        training_data = training_data[self.feature_list]
        scaler = MinMaxScaler()
        self.gpr = fit_gpr(scaler.fit_transform(training_data), training_labels)

    def train_poly_reg(self, training_data: pd.DataFrame, training_labels: pd.Series, memory_path=None):
        training_data = training_data[self.feature_list]
        self.poly_reg = fit_poly_reg(training_data, training_labels, memory_path=memory_path)


    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            result[bearing] = pd.Series(flatten_predictions(self.prediction_model.predict(df[self.feature_list])),
                                        name="rul_predictions")
        return result

    def predict_svr(self, df_dict: Dict[str, pd.DataFrame]):
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            df_tmp = df[self.feature_list]
            result[bearing] = pd.Series(self.svr.predict(df_tmp),
                                        name="rul_predictions")
        return result

    def predict_gpr(self, df_dict: Dict[str, pd.DataFrame]):
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            df_tmp = df[self.feature_list]
            scaler = MinMaxScaler()
            df_tmp = scaler.fit_transform(df_tmp)
            result[bearing] = pd.Series(self.gpr.predict(df_tmp),
                                        name="rul_predictions")
        return result

    def predict_poly_reg(self, df_dict: Dict[str, pd.DataFrame]):
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            df_tmp = df[self.feature_list]
            result[bearing] = pd.Series(self.poly_reg.predict(df_tmp),
                                        name="rul_predictions")
        return result

    def __str__(self):
        return "Features: " + str(self.feature_list) + " FFNN rul prediction."
