import sys
sys.path.append("./")

import numpy as np
import pandas as pd
from tensorflow.keras import layers, Input, Model
from tensorflow import keras
from rul_features.learned_features.Embedder import Embedding
from util.constants import SPECTRA_SHAPE, SPECTRA_CSV_NAME, LEARNING_SET, FULL_TEST_SET, CNN_PATH
from pre_processing.features import read_feature_dfs_as_dict, df_dict_to_df_dataframe

class CNNEmbedding(Embedding):
    def fit_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        pass

    def embed_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cnn_embedding_layers: Model = keras.models.load_model(CNN_PATH, compile=False)
        df = df.to_numpy()
        df = np.array([row.reshape(SPECTRA_SHAPE) for row in df])
        return pd.DataFrame(cnn_embedding_layers.predict(df))

    def fit_frequency_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        print("CNNEmbedding does not support extracting frequency from time-frequency spectrum")
        assert False

    def embed_frequency_df(self, df: pd.DataFrame):
        print("CNNEmbedding does not support extracting frequency from time-frequency spectrum")
        assert False


def build_multiscale_cnn(input_shape: tuple) -> (Model, Model):
    input_img = Input(shape=input_shape)
    conv_1 = layers.Conv2D(10, (6, 6))(input_img)
    max_pool_1 = layers.MaxPool2D((2, 2))(conv_1)
    max_pool_1 = layers.BatchNormalization()(max_pool_1)
    conv_2 = layers.Conv2D(10, (6, 6))(max_pool_1)
    max_pool_2 = layers.MaxPool2D((2, 2))(conv_2)
    max_pool_2 = layers.BatchNormalization()(max_pool_2)
    cnn = layers.Flatten()(max_pool_2)

    ffnn = layers.Dense(512, activation='relu')(cnn)
    ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(512, activation='relu')(ffnn)
    ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(1)(ffnn)

    return Model(input_img, ffnn), Model(input_img, cnn)


def train_and_store_cnn():
    spectra_training_df, spectra_training_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_sub_set=LEARNING_SET, csv_name=SPECTRA_CSV_NAME)
    )
    spectra_validation_df, spectra_validation_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_sub_set=FULL_TEST_SET, csv_name=SPECTRA_CSV_NAME)
    )

    spectra_training_dfs = spectra_training_df.to_numpy()
    spectra_training_dfs = np.array([row.reshape(SPECTRA_SHAPE) for row in spectra_training_dfs])

    spectra_validation_dfs = spectra_validation_df.to_numpy()
    spectra_validation_dfs = np.array([row.reshape(SPECTRA_SHAPE) for row in spectra_validation_dfs])

    full_cnn, cnn_embedding_layers = build_multiscale_cnn(SPECTRA_SHAPE)
    full_cnn.compile(optimizer='adam', loss='mean_squared_error')
    _ = full_cnn.fit(x=spectra_training_dfs, y=spectra_training_labels,
                     epochs=20, verbose=0,
                     validation_data=(spectra_validation_dfs, spectra_validation_labels))

    cnn_embedding_layers.save(CNN_PATH)


if __name__ == '__main__':
    print('Train and store cnn')
    train_and_store_cnn()
