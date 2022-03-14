import predictor
import streamlit as st
import pickle
from datetime import date

import yfinance as yf
from plotly import graph_objs as go

START = "2012-01-01"
END = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    return data


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def main():

    st.title("MULTIPLE STOCK PRICE PREDICTOR")

    html_paragraph = '<p style="font-size: 25px"> Welcome to the stock market predictor!You can use this to visualize the stock market data in order to conduct your own analysis on them, or use it to predict the future stock price of a selected stock.<br> Please note that this isn\'t meant to be financial advice.</p>'
    st.markdown(html_paragraph, unsafe_allow_html=True)

    option = st.selectbox("What would you like to do today?", ("SELECT AN OPTION", "VIEW STOCK MARKET DATA", "PREDICT FUTURE STOCK PRICE"))

    if option == 'SELECT AN OPTION':
        pass

    else:
        stocks = ("GOOG", "AAPL", "MSFT", "GME")
        selected_stock = st.selectbox("Select stock for prediction", stocks)

        data = load_data(selected_stock)

        if option == 'VIEW STOCK MARKET DATA':
            st.subheader(f'STATISTICS OF {selected_stock}')
            st.write(data.tail())
            plot_raw_data(data)

        if option == 'PREDICT FUTURE STOCK PRICE':
            st.write('SNEAK PEEK OF THE DATA')
            st.write(data.tail())

            try:
                pickle_model = pickle.load(open(f'Pickle_files/{selected_stock}', 'rb'))
                predictor.test_model(pickle_model, selected_stock)

            except FileNotFoundError:
                model = predictor.lstm_model(data, selected_stock)
                predictor.test_model(model, selected_stock)


main()
