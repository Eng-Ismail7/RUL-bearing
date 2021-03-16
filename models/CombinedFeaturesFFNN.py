from typing import Dict
import pandas as pd
from tensorflow.keras.callbacks import History

from models.DegradationModel import DegradationModel
from models.DataSetType import DataSetType
from rul_features.learned_features.Embedder import Embedding
from util.helper import flatten_predictions
from rul_prediction.ffnn import fit_ffnn
from rul_prediction.suport_vector_regression import fit_svr
from rul_prediction.gpr import fit_gpr
from rul_prediction.poly_reg import fit_poly_reg

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class EmbeddingFeaturesFNNN(DegradationModel):
    def __init__(self, name: str, embedding_method: Embedding, encoding_size: int, data_set_type: DataSetType,
                 use_frequency_embedding:bool = False):
        DegradationModel.__init__(self, name=name, data_set_type=data_set_type)
        self.combiner = embedding_method
        self.encoding_size = encoding_size
        self.use_frequency_embedding = use_frequency_embedding

    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            df = self._embed_data(df)
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            result[bearing] = pd.Series(
                flatten_predictions(self.prediction_model.predict(df)),
                name="rul_predictions")
        return result

    def predict_svr(self, df_dict: Dict[str, pd.DataFrame]):
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            df = self._embed_data(df)
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            result[bearing] = pd.Series(
                self.svr.predict(df),
                name="rul_predictions")
        return result

    def predict_gpr(self, df_dict: Dict[str, pd.DataFrame]):
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            df = self._embed_data(df)
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            result[bearing] = pd.Series(
                self.gpr.predict(df),
                name="rul_predictions")
        return result

    def predict_poly_reg(self, df_dict: Dict[str, pd.DataFrame]):
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            result[bearing] = pd.Series(
                self.poly_reg.predict(self._embed_data(df)),
                name="rul_predictions")
        return result

    def train(self, training_data: pd.DataFrame, training_labels: pd.Series, validation_data: pd.DataFrame,
              validation_labels: pd.Series) -> History:
        training_data = self._fit_and_embed_training_data(training_data)   
        scaler = MinMaxScaler()
        training_data = pd.DataFrame(scaler.fit_transform(training_data), columns=training_data.columns)
        if validation_data is not None:
            validation_data = self._embed_data(validation_data)
        if validation_data is not None:
            self.prediction_model, self.trainings_history = fit_ffnn(X=training_data, y=training_labels,
                                                                     dropout=True,
                                                                     validation_data=(
                                                                     validation_data, validation_labels))
        else:
            self.prediction_model, self.trainings_history = fit_ffnn(X=training_data, y=training_labels,
                                                                     dropout=True,
                                                                     validation_data=None)
        return self.trainings_history

    def train_svr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        training_data = self._fit_and_embed_training_data(training_data)
        scaler = MinMaxScaler()
        training_data = pd.DataFrame(scaler.fit_transform(training_data), columns=training_data.columns)
        self.svr = fit_svr(training_data, training_labels)

    def train_gpr(self, training_data: pd.DataFrame, training_labels: pd.Series):
        training_data = self._fit_and_embed_training_data(training_data)
        scaler = MinMaxScaler()
        training_data = pd.DataFrame(scaler.fit_transform(training_data), columns=training_data.columns)
        self.gpr = fit_gpr(training_data, training_labels)

    def train_poly_reg(self, training_data: pd.DataFrame, training_labels: pd.Series, memory_path=None):
        training_data = self._fit_and_embed_training_data(training_data)
        self.poly_reg = fit_poly_reg(training_data, training_labels, memory_path=memory_path)

    def _fit_and_embed_training_data(self, df:pd.DataFrame)-> pd.DataFrame:
        if self.use_frequency_embedding:
            self.combiner.fit_frequency_embedding(df, encoding_size=self.encoding_size, verbose=False)
            training_data = self.combiner.embed_frequency_df(df)
        else:
            self.combiner.fit_embedding(df, encoding_size=self.encoding_size, verbose=False)
            training_data = self.combiner.embed_data_frame(df)
        return training_data

    def _embed_data(self, df: pd.DataFrame):
        if self.use_frequency_embedding:
            return self.combiner.embed_frequency_df(df)
        else:
            return self.combiner.embed_data_frame(df)

