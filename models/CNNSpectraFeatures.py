from typing import Dict
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import History

from pre_processing.features import df_dict_to_df_dataframe
from util.constants import SPECTRA_SHAPE
from util.helper import flatten_predictions

from rul_prediction.cnn import fit_cnn
from models.DegradationModel import DegradationModel
from models.DataSetType import DataSetType


class CNNSpectraFeatures(DegradationModel):
    def __init__(self, name):
        DegradationModel.__init__(self, name, DataSetType.spectra)

    def train(self, training_data: pd.DataFrame, training_labels: pd.Series, validation_data: pd.DataFrame,
              validation_labels: pd.Series) -> History:
        spectra_dfs = training_data.to_numpy()
        spectra_dfs = np.array([row.reshape(SPECTRA_SHAPE) for row in spectra_dfs])
        validation_data = validation_data.to_numpy()
        validation_data = np.array([row.reshape(SPECTRA_SHAPE) for row in validation_data])

        self.prediction_model, self.trainings_history = fit_cnn(spectra_dfs, training_labels,
                                                                input_shape=SPECTRA_SHAPE,
                                                                validation_data=(validation_data, validation_labels))
        return self.trainings_history

    def train_svr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        print("CNN does not support SVR.")
        assert False

    def train_gpr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        print("CNN does not support SVR.")
        assert False

    def train_poly_reg(self, training_data: pd.DataFrame, training_labels: pd.Series, memory_path=None):
        print("CNN does not support SVR.")
        assert False

    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        df_dict = self._reformat_spectra(df_dict)
        result = {}
        for bearing, df in df_dict.items():
            result[bearing] = pd.Series(flatten_predictions(self.prediction_model.predict(df)),
                                        name="rul_predictions")
        return result

    def predict_svr(self, df_dict: Dict[str, pd.DataFrame]):
        print("CNN does not support SVR.")
        assert False

    def predict_gpr(self, df_dict: Dict[str, pd.DataFrame]):
        print("CNN does not support SVR.")
        assert False

    def predict_poly_reg(self, df_dict: Dict[str, pd.DataFrame]):
        print("CNN does not support SVR.")
        assert False

    def _reformat_spectra(self, df_dict: Dict[str, pd.DataFrame]):
        return {bearing: df.to_numpy().reshape((df.shape[0], SPECTRA_SHAPE[0], SPECTRA_SHAPE[1], SPECTRA_SHAPE[2])) for
                bearing, df in df_dict.items()}
