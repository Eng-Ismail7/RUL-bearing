import pandas as pd
import matplotlib.pyplot as plt
import sys

from rul_features.learned_features.Embedder import Embedding

# keras imports
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow import keras

def build_contractive_denoising_auto_encoder():
    # Nice to have
    pass


def build_deep_autoencoder(input_dim: int, encoding_dim: int):
    input_df = Input(input_dim)
    
    encoded = Dense(160)(input_df)
    encoded = Dense(80)(encoded)
    
    encoded = Dense(encoding_dim)(encoded)

    decoded = Dense(80)(encoded)
    decoded = Dense(160)(decoded)
    decoded = Dense(input_dim)(decoded)

    return Model(input_df, decoded), Model(input_df, encoded)


def build_sparse_autoencoder(input_dim: int, encoding_dim: int) -> (Model, Model):
    input_df = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_df)
    decoded = Dense(input_dim)(encoded)

    return Model(input_df, decoded), Model(input_df, encoded)


def fit_autoencoder(trainings_data: pd.DataFrame, encoding_size: int, epochs: int = 50, verbose: bool = False,
                    validation_data: tuple = None) -> (
        Model, Model):

    input_dim = len(trainings_data.keys())
    autoencoder_model, encoder_model = build_deep_autoencoder(input_dim=input_dim, encoding_dim=encoding_size)

    opt = keras.optimizers.Adam(learning_rate=0.01)
    autoencoder_model.compile(optimizer=opt, loss='mean_squared_error')

    normalized_trainings_data = (trainings_data-trainings_data.mean())/trainings_data.std()

    history = autoencoder_model.fit(normalized_trainings_data, normalized_trainings_data,
                                    epochs=epochs,
                                    batch_size=256,
                                    shuffle=True,
                                    validation_data=validation_data)

    if verbose:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.figure(figsize=(5, 5))
        plt.rc("errorbar", capsize=5)
        plt.tight_layout()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Autoencoder loss')
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend(["Learning", "Testing"])
        plt.show()

    return autoencoder_model, encoder_model


def autoencoder_embedded_data_frame(bearing_df: pd.DataFrame, n_components: int, verbose: bool = False):
    autoencoder, encoder = fit_autoencoder(bearing_df, encoding_size=n_components, verbose=verbose)
    transformed_df = encoder.predict(bearing_df)
    return pd.DataFrame(transformed_df), encoder


class AutoencoderEmbedding(Embedding):
    def fit_embedding(self, df: pd.DataFrame, encoding_size: int, verbose: bool):
        _, self.embedder = fit_autoencoder(df, encoding_size=encoding_size, verbose=verbose, epochs=20)

    def embed_data_frame(self, df: pd.DataFrame):
        return pd.DataFrame(self.embedder.predict(df))
