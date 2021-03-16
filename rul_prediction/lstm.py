import pandas as pd
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import History


def build_lstm_model(input_shape: tuple, dropout: bool, lstm_layers: int, lstm_units: int) -> Model:
    input_df = Input(shape=input_shape)

    lstm = layers.LSTM(lstm_units)(input_df)
    if dropout:
        lstm = layers.Dropout(0.5)(lstm)
    for i in range(lstm_layers):
        lstm = layers.Dense(lstm_units)(lstm)
        if dropout:
            lstm = layers.Dropout(0.5)(lstm)
    lstm = layers.Dense(1)(lstm)

    return Model(input_df, lstm)


def fit_lstm(X: list, y: list, dropout: bool = False, epochs: int = 250, hidden_layers: int = 2,
             hidden_units=128) -> (History, Model):
    input_shape = (300, len(X[0].keys()))
    lstm = build_lstm_model(input_shape, dropout, lstm_layers=hidden_layers, lstm_units=hidden_units)
    lstm.compile(optimizer='adam', loss='mean_absolute_error')
    training_history = lstm.fit(x=X, y=y, epochs=epochs, shuffle=True)
    return lstm, training_history
