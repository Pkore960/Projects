# app.py

import streamlit as st
from Stock_Price_Predictor import Model  # Assuming Model.py is in Stock_Price_Predictor folder
import time
import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
from plotly import graph_objs as go
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras.layers import Dropout, LSTM, Dense
from keras.models import Sequential

cf.go_offline()
plt.style.use('seaborn-darkgrid')

def load_data(ticker, start_date, today_date):
    data = yf.download(ticker, start_date, today_date)
    data.reset_index(inplace=True)
    return data

def main():
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    TODAY = date.today().strftime("%Y-%m-%d")

    st.title('Stock Price & Trend Prediction')

    progress = st.progress(0)
    for i in range(100):
        progress.progress(i + 1)
    st.balloons()

    st.write("""
    ### Explore different classifiers
    Which one is best?
    """)

    classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM", "KNN", "Linear Regression", "LSTM", "Random Forest"))

    START = st.sidebar.date_input("Enter a date")

    market = ('US Stocks Market', 'Indian Stocks Market')
    market_type = st.sidebar.selectbox('Select Market Type', market)
    st.header('')

    if market_type == 'US Stocks Market':
        curr = 'in USD'
    elif market_type == 'Indian Stocks Market':
        curr = 'in INR'

    user_input = st.sidebar.text_input('Enter Stock Ticker (GOOG, AAPL, MSFT, GME, TSLA, M&M.NS, VBL.NS)', 'GOOG')

    ok = st.sidebar.button("Predict")

    if ok:
        with st.spinner('Loading data...'):
            df = load_data(user_input, START, TODAY)

        st.subheader(f'{user_input} Data from {START} to Today')
        st.write(df.describe())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low

        # Stock Trend Prediction
        st.subheader('Time Series Graph')
        fig = df[['Close', 'Open-Close', 'High-Low']].iplot(kind='lines', xTitle='Time (Days)',
                                                             yTitle=f'Price  ({curr})', asFigure=True)
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

        st.subheader('Closing Price vs Time chart with 3 Months Moving Avg.')
        ma100 = df.Close.rolling(90).mean()
        fig = plt.figure(figsize=(12, 7))
        plt.plot(ma100, 'g', label='3 Months MA')
        plt.plot(df.Close)
        plt.xlabel('Time (Days)', fontsize='15')
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
        plt.xlabel('Time (Days)', fontsize='15')
        plt.ylabel(f'Price  ({curr})', fontsize='15')
        plt.legend()
        st.pyplot(fig)

        # Predicting Stock Price Prediction
        X = df[['Open-Close', 'High-Low']]
        Y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Trading Signal

        split_percentage = 0.7
        split = int(split_percentage * len(df))

        x_train = X[:split]
        y_train = Y[:split]

        x_test = X[split:]
        y_test = Y[split:]

        st.subheader(f'Prediction Using {classifier_name}')

        def get_classifier(clf_name):
            if clf_name == "SVM":
                Model = SVC().fit(x_train, y_train)
            elif clf_name == "Linear Regression":
                Model = LinearRegression()
                Model.fit(x_train, y_train)
            elif clf_name == "Random Forest":
                Model = RandomForestRegressor(n_estimators=500, random_state=42,
