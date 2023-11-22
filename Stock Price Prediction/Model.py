import streamlit as st

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.layers import Dropout,LSTM,Dense
from keras.models import Sequential
from plotly import graph_objs as go
from datetime import date

import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cufflinks as cf
cf.go_offline()
plt.style.use('seaborn-darkgrid')

progress=st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price & Trend Prediction')

progress=st.progress(0)
for i in range(100):
    # time.sleep(0.01)
    progress.progress(i+1)
# st.balloons()

st.write("""
### Explore different classifiers
Which one is best?
""")

classifier_name=st.sidebar.selectbox("Select Classifier",("SVM","KNN","Linear Regression","LSTM","Random Forest"))

START = st.sidebar.date_input("Enter a date")

market = ('US Stocks Market', 'Indian Stocks Market')
market_type = st.sidebar.selectbox('Select Market Type', market)
st.header('')

if market_type == 'US Stocks Market':
    curr = 'in USD'
elif market_type == 'Indian Stocks Market':
    curr = 'in INR'

user_input = st.sidebar.text_input('Enter Stock Ticker (GOOG,AAPL, MSFT, GME, TSLA, M&M.NS, VBL.NS)', 'GOOG')

ok = st.sidebar.button("Predict")

if ok:
    with st.spinner('Loading data...'):
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        df = load_data(user_input)
    df.set_index("Date", inplace=True)
    df.dropna(inplace=True)
 
    st.subheader(f'{user_input} Data from {START} to {TODAY}')
    st.write(df.describe())
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    first_day_closing_price = df.iloc[0]['Close']


    # Stock Trend Prediction
    st.subheader('Time Series Graph')
    fig = go.Figure()
    fig = df[['Close', 'Open-Close', 'High-Low']].iplot(kind='lines',xTitle='Year/Date/Time', yTitle=f'Price  ({curr})', asFigure=True)
    fig.layout.update(title_text='Time Series Graph',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    st.write("")
    st.subheader('Closing Price vs Time chart with 3 Months Moving Avg.')
    ma100 = df.Close.rolling(90).mean()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(ma100, 'g', label='3 Months MA')
    plt.plot(df.Close)
    plt.xlabel('Year', fontsize='15')
    plt.ylabel(f'Price  ({curr})', fontsize='15')
    plt.legend()
    st.pyplot(fig)

    st.write("")
    st.subheader('Stock Trend Prediction and Buy/Sell Strategy')
    st.markdown("If the Green line crosses Red line and goes **UP** then there is <span style='color:green;'>UP TREND</span> and Put <span style='color:red;'>BUY Order</span>.", unsafe_allow_html=True)
    st.markdown("If the Green line crosses Red line and goes **DOWN** then there is <span style='color:green;'>DOWN TREND</span> and Put <span style='color:red;'>SELL Order</span>.", unsafe_allow_html=True)
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(ma100, 'g', label='3 Months MA')
    plt.plot(ma200, 'r', label='6 Months MA')
    plt.plot(df.Close)
    plt.xlabel('Year', fontsize='15')
    plt.ylabel(f'Price  ({curr})', fontsize='15')
    plt.legend()
    st.pyplot(fig)

    # Predicting Stock Price Prediction
    X = df[['Open-Close', 'High-Low']]
    Y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Treading Signal

    split_percentage = 0.7
    split = int(split_percentage*len(df))

    x_train = X[:split]
    y_train = Y[:split]

    x_test = X[split:]
    y_test = Y[split:]


    st.write("")
    st.subheader(f'Predicting Stock Price Direction Using {classifier_name}')
    def get_classifier(clf_name):
        if clf_name=="SVM":
            Model = SVC().fit(x_train, y_train)
        elif clf_name=="Linear Regression":
            Model = LinearRegression()
            Model.fit(x_train, y_train)
        elif clf_name=="Random Forest":
            Model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=5, bootstrap=True)
            Model.fit(x_train, y_train)
        elif clf_name=="LSTM":
            Model=Sequential()
            Model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
            Model.add(LSTM(units=50))
            Model.add(Dense(1))            
            Model.compile(optimizer='adam',loss='mean_squared_error')
            Model.fit(x_train, y_train, epochs=50, batch_size=32)  
        else:
            Model = KNeighborsClassifier(n_neighbors=100)
            Model.fit(x_train, y_train)
        return Model
    
    Model=get_classifier(classifier_name)
    df['Return'] = df.Close.pct_change()
    df['Cum_Return'] = df['Return'].cumsum()
    df['Predicted_Signal'] = Model.predict(X)
    df['Strategy'] = df.Return * df.Predicted_Signal.shift(1)
    df['Strategy_Return'] = df['Strategy'].cumsum()
    fig = df[['Strategy_Return','Cum_Return']].iplot(kind='lines',xTitle='Year/Date/Time', yTitle='', asFigure=True)
    fig.layout.update(title_text='Prediction',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
    y_pred = Model.predict(X)
    st.success(f"Classifier = {classifier_name}")

    if classifier_name not in ["Random Forest", "Linear Regression", "LSTM"]:
        acc = accuracy_score(Y, y_pred)*100
        st.success(f"Accuracy = {acc:.2f}%")
    




    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
    scaler=MinMaxScaler(feature_range=(0,1))
    data_training_array=scaler.fit_transform(data_training)
    #Splitting data into x_train and y_train
    x_train=[]
    y_train=[]
    for i in range(100,data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])
    x_train,y_train=np.array(x_train),np.array(y_train)

    model=get_classifier("LSTM")
    model.fit(x_train, y_train)
    
    #Testing Part
    past_100_days=data_training.tail(100)

    final_df=past_100_days.append(data_testing,ignore_index=True)
    input_data=scaler.fit_transform(final_df)
    x_test=[]
    y_test=[]

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test=np.array(x_test),np.array(y_test)
    y_predicted=model.predict(x_test)
    # y_predicted= scaler.inverse_transform(y_predicted)
    scale=scaler.scale_
    scale_factor=1/(scale[0])
    y_predicted=y_predicted*scale_factor
    y_test=y_test*scale_factor

    #Final Graph
    st.write("")
    st.subheader('Predictions vs Original of Testing Data')
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test,label='Original Price')
    plt.plot(y_predicted,label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    # st.success(f"Classifier = LSTM")
    # acc = accuracy_score(y_test, y_predicted)*100
    # st.success(f"Accuracy = {acc:.2f}%")