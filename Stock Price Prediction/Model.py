import streamlit as st
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_scores
from pyexpat import features
import yfinance as yf
from datetime import date
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price & Trend Prediction')

market = ('US Stocks Market', 'Indian Stocks Market')
market_type = st.selectbox('Select Market Type', market)
st.header('')

if market_type == 'US Stocks Market':
    curr = 'in USD'
elif market_type == 'Indian Stocks Market':
    curr = 'in INR'

user_input = st.text_input('Enter Stock Ticker (GOOG,AAPL, MSFT, GME, TSLA, M&M.NS, VBL.NS)', 'GOOG')

# n_years = st.slider('Years of prediction:', 1, 4)
# period = n_years * 365

ok = st.button("Predict")
if ok:
    df = yf.download(user_input, START, TODAY)
    df.reset_index(inplace=True)

    st.subheader(f'{user_input} Data from 2015 to Today')
    st.write(df.describe())
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    # st.write(df)

    fig = go.Figure()
    fig.layout.update(title_text='Time Series Data',
                      xaxis_rangeslider_visible=True)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
    st.plotly_chart(fig)

    # Stock Trend Prediction
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 7))
    plt.plot(df.Close)
    plt.xlabel('Time', fontsize='15')
    plt.ylabel(f'Price  ({curr})', fontsize='15')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 3 Months Moving Avg.')
    ma100 = df.Close.rolling(90).mean()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(ma100, 'g', label='3 Months MA')
    plt.plot(df.Close)
    plt.xlabel('Time', fontsize='15')
    plt.ylabel(f'Price  ({curr})', fontsize='15')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 6 Months Moving Avg.')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(ma100, 'g', label='3 Months MA')
    plt.plot(ma200, 'r', label='6 Months MA')
    plt.plot(df.Close)
    plt.xlabel('Time', fontsize='15')
    plt.ylabel(f'Price  ({curr})', fontsize='15')
    plt.legend()
    st.pyplot(fig)

    # Predicting Stock Price Prediction using Support Vector Machines(Supervised Machine Learning)
    X = df[['Open-Close', 'High-Low']]
    Y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Treading Signal

    split_percentage = 0.7
    split = int(split_percentage*len(df))

    x_train = X[:split]
    y_train = Y[:split]

    x_test = X[split:]
    y_test = Y[split:]

    Model = SVC().fit(x_train, y_train)

    # for i in range(df.Close.shape[0],period):
    #     df.Close.append(df.Close[df.Close.shape[0]-1])

    df['Return'] = df.Close.pct_change()
    df['Cum_Ret'] = df['Return'].cumsum()
    df['Predicted_Signal'] = Model.predict(X)
    df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

    st.subheader('Predictions vs Original')
    fig = plt.figure(figsize=(15, 9))
    plt.plot(df['Cum_Ret'], label='Original Price')
    plt.plot(df['Cum_Strategy'], 'r', label='Predicted Price')
    plt.xlabel('Time', fontsize='15')
    plt.ylabel('Price', fontsize='15')
    plt.legend()
    st.pyplot(fig)
