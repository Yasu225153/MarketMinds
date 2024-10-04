# **************** IMPORT PACKAGES ********************
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.linear_model import LinearRegression

# **************** ARIMA MODEL **************************
def run_arima(df):
    uniqueVals = df["Code"].unique()
    df = df.reset_index()  
    df = df.set_index("Code")

    def arima_model(train, test):
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            history.append(test[t])
        return predictions

    for company in uniqueVals[:10]:
        data = df.loc[company, :].reset_index()
        data['Price'] = data['Close']
        Quantity_date = data[['Price', 'Date']].set_index('Date')
        Quantity_date['Price'] = Quantity_date['Price'].astype(float).fillna(method='bfill')

        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(Quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)

        quantity = Quantity_date['Price'].values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]

        predictions = arima_model(train, test)

        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ARIMA.png')
        plt.close(fig)

        arima_pred = predictions[-1]
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        return arima_pred, error_arima

# **************** LSTM MODEL **************************
def run_lstm(df):
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]

    training_set = df.iloc[:, 4:5].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_forecast = np.append(X_train[-1, 1:], y_train[-1])
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=25, batch_size=32)

    real_stock_price = dataset_test.iloc[:, 4:5].values

    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values.reshape(-1, 1)

    testing_set = sc.transform(testing_set)

    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i-7:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/LSTM.png')
    plt.close(fig)

    # Predict the next stock price using the reshaped X_forecast
    predicted_next_price = sc.inverse_transform(regressor.predict(X_forecast))

    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    return predicted_next_price[-1, 0], error_lstm

# def run_lstm(df):
#     dataset_train = df.iloc[0:int(0.8*len(df)), :]
#     dataset_test = df.iloc[int(0.8*len(df)):, :]

#     training_set = df.iloc[:, 4:5].values

#     sc = MinMaxScaler(feature_range=(0, 1))
#     training_set_scaled = sc.fit_transform(training_set)

#     X_train, y_train = [], []
#     for i in range(7, len(training_set_scaled)):
#         X_train.append(training_set_scaled[i-7:i, 0])
#         y_train.append(training_set_scaled[i, 0])

#     X_train, y_train = np.array(X_train), np.array(y_train)
#     X_forecast = np.append(X_train[-1, 1:], y_train[-1])

#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

#     regressor = Sequential()

#     regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     regressor.add(Dropout(0.1))

#     regressor.add(LSTM(units=50, return_sequences=True))
#     regressor.add(Dropout(0.1))

#     regressor.add(LSTM(units=50, return_sequences=True))
#     regressor.add(Dropout(0.1))

#     regressor.add(LSTM(units=50))
#     regressor.add(Dropout(0.1))

#     regressor.add(Dense(units=1))

#     regressor.compile(optimizer='adam', loss='mean_squared_error')

#     regressor.fit(X_train, y_train, epochs=25, batch_size=32)

#     real_stock_price = dataset_test.iloc[:, 4:5].values

#     dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
#     testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values.reshape(-1, 1)

#     testing_set = sc.transform(testing_set)

#     X_test = []
#     for i in range(7, len(testing_set)):
#         X_test.append(testing_set[i-7:i, 0])

#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     predicted_stock_price = regressor.predict(X_test)
#     predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#     fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
#     plt.plot(real_stock_price, label='Actual Price')
#     plt.plot(predicted_stock_price, label='Predicted Price')
#     plt.legend(loc=4)
#     plt.savefig('static/LSTM.png')
#     plt.close(fig)

#     predicted_stock_price = sc.inverse_transform(regressor.predict(X_forecast))

#     error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

#     return predicted_stock_price[-1, 0], error_lstm

# **************** LINEAR REGRESSION MODEL **************************
def run_lr(df):
    df = df[['Close']].copy()
    df['Prediction'] = df['Close'].shift(-7)

    # Corrected the drop() method call
    X = np.array(df.drop(['Prediction'], axis=1))[:-7]
    y = np.array(df['Prediction'])[:-7]

    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-7:]

    lr = LinearRegression()
    lr.fit(X, y)

    lr_confidence = lr.score(X, y)

    lr_prediction = lr.predict(x_forecast)

    lr_pred = lr_prediction[-1]

    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    predictions = lr_prediction
    real_stock_price = df['Close'][-7:]
    plt.plot(real_stock_price.index, real_stock_price, label='Actual Price')
    plt.plot(real_stock_price.index, predictions, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig('static/LR.png')
    plt.close(fig)

    error_lr = math.sqrt(mean_squared_error(real_stock_price, lr_prediction))

    return lr_pred, error_lr

