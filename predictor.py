import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle

from datetime import date
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from plotly import graph_objs as go


def predict(data):
    st.text('Data head ....')
    st.write(data.head())


# Prepare data
def pre_process(data):
    prediction_days = 60
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days: x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def lstm_model(data, ticker):
    x_train, y_train = pre_process(data)
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Prediction
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    pickle.dump(model, open(f'{ticker}', 'wb'))

    return model


def test_model(model, ticker):
    prediction_days = 60
    scaler = MinMaxScaler(feature_range=(0, 1))

    test_data = load_test_data(ticker)
    test_data.reset_index(inplace=True)

    st.write("Test data head")
    st.write(test_data.head())

    actual_price_model = pd.DataFrame(columns=['Date', 'Close'])
    predicted_price_model = pd.DataFrame(columns=['Date', 'Close'])

    model_input = test_data['Close'].values
    model_input = model_input.reshape(-1, 1)

    model_input = scaler.fit_transform(model_input)

   # Predictions on test_data
    x_test = []
    for x in range(prediction_days, len(model_input)):
        row = []
        x_test.append(model_input[x - prediction_days: x, 0])
        row.append(test_data['Date'][x])
        row.append(test_data['Close'][x])
        actual_price_model.loc[len(actual_price_model.index)] = row
        predicted_price_model.loc[len(predicted_price_model.index)] = [test_data['Date'][x], 0]

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    predicted_price_model['Close'] = predicted_prices

    st.write('Actual prices')
    st.write(actual_price_model)
    st.write('Predicted prices')
    st.write(predicted_price_model)

    plot_predictions(actual_price_model, predicted_price_model, ticker)


def load_test_data(ticker):
    return yf.download(ticker, "2020-01-01", date.today().strftime("%Y-%m-%d"))


def plot_predictions(actual_price_model, predicted_price_model, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_price_model['Date'], y=actual_price_model['Close'], name='actual_price'))
    fig.add_trace(go.Scatter(x=predicted_price_model['Date'], y=predicted_price_model['Close'], name='predicted_price'))
    fig.layout.update(title_text=f'{ticker} share price')
    st.plotly_chart(fig)