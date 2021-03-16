import pandas as pd
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import SGD, Adam
import sys
from tensorflow import keras

def build_ffnn_model_for_cnn(input_shape: tuple, dropout: bool) -> Model:
    input_df = Input(shape=input_shape)

    ffnn = layers.Dense(128, activation='relu')(input_df)
    if dropout:
        ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(64, activation='relu')(ffnn)
    if dropout:
        ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(1)(ffnn)

    return Model(input_df, ffnn)


def build_ffnn_model(input_shape: tuple, dropout: bool, hidden_layers: int, hidden_units: int) -> Model:
    input_df = Input(shape=input_shape)

    ffnn = layers.Dense(hidden_units, activation='relu')(input_df)
    if dropout:
        ffnn = layers.Dropout(0.5)(ffnn)

    for i in range(hidden_layers - 1):
        ffnn = layers.Dense(hidden_units, activation='relu')(ffnn)
        if dropout:
            ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(1)(ffnn)

    return Model(input_df, ffnn)


def fit_ffnn(X: pd.DataFrame, y: pd.Series, dropout: bool = True, epochs: int = 50, hidden_layers: int = 4,
             hidden_units=512, validation_data=(None, None)) -> (History, Model):
    input_shape = (len(X.keys()),)
    ffnn = build_ffnn_model(input_shape, dropout, hidden_layers=hidden_layers, hidden_units=hidden_units)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    ffnn.compile(optimizer=opt, loss='mean_squared_error')
    if validation_data is (None, None):
        training_history = ffnn.fit(x=X, y=y, epochs=epochs, shuffle=True, verbose=0)
    else:
        training_history = ffnn.fit(x=X, y=y, epochs=epochs, shuffle=True, verbose=0)
    return ffnn, training_history


if __name__ == '__main__':
    mod = build_ffnn_model((5,), True, 2, 512)