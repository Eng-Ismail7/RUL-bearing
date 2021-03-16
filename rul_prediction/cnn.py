import pandas as pd
import numpy as np
from tensorflow.keras import layers, Input, Model


def build_multiscale_cnn(input_shape: tuple) -> Model:
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

    return Model(input_img, ffnn)


def fit_cnn(X: np.array, y: pd.Series, epochs: int = 20, input_shape=(129, 21, 2),
            validation_data=None):
    cnn_model = build_multiscale_cnn(input_shape)
    cnn_model.compile(optimizer='adam', loss='mean_squared_error')
    if validation_data is None:
        training_history = cnn_model.fit(x=X, y=y, epochs=epochs, verbose=0)
    else:
        training_history = cnn_model.fit(x=X, y=y, epochs=epochs, verbose=0, validation_data=validation_data)
    return cnn_model, training_history


if __name__ == '__main__':
    mod = build_multiscale_cnn((129, 21, 2))
"""
REN ET AL. 2018 FOR FEATURE EXTRACTION e.g. FREQUENCY IMAGE Generation based non FFT
ZHU ET AL. 2019 for multisclae CNN applied to PRONOSTIA
DING ET AL. 2017 for multiscale CNN architecture and WT based Image
CHEN ET AL. 2020 Another possible Frequency Image?
"""
