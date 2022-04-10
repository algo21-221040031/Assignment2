import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics
from DataProcess import *

if __name__ == "__main__":

    data = pd.read_csv("../Data/601988.SH.csv")
    test_set2 = data.loc[3501:, :]
    data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data.drop(['ts_code', 'trade_date'], axis=1)
    data = pd.DataFrame(data, dtype=np.float64)

    # 划分训练集和测试集
    trainingSet = data.loc['2007-01-04':'2021-06-21', :]  # 3501
    testSet = data.loc['2021-06-22':, :]  # 180

    plt.figure(figsize=(10, 6))
    plt.plot(trainingSet['close'], label='training_set')
    plt.plot(testSet['close'], label='test_set')
    plt.title('Close price')
    plt.xlabel('time', fontsize=12, verticalalignment='top')
    plt.ylabel('close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()

    temp = np.array(trainingSet['close'])

    # 一阶差分
    trainingSet['diff_1'] = trainingSet['close'].diff(1)
    plt.figure(figsize=(10, 6))
    trainingSet['diff_1'].plot()
    plt.title('First-order diff')
    plt.xlabel('time', fontsize=12, verticalalignment='top')
    plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
    plt.show()

    # 二阶差分
    trainingSet['diff_2'] = trainingSet['diff_1'].diff(1)
    plt.figure(figsize=(10, 6))
    trainingSet['diff_2'].plot()
    plt.title('Second-order diff')
    plt.xlabel('time', fontsize=12, verticalalignment='top')
    plt.ylabel('diff_2', fontsize=14, horizontalalignment='center')
    plt.show()

    temp1 = np.diff(trainingSet['close'], n=1)

    # 白噪声检验
    training_data1 = trainingSet['close'].diff(1)
    temp2 = np.diff(trainingSet['close'], n=1)
    print(acorr_ljungbox(temp2, lags=2, boxpierce=True))

    AcfPacfPlot(trainingSet['close'],acf_lags=160)

    price = list(temp2)
    data2 = {
        'trade_date': trainingSet['diff_1'].index[1:],
        'close': price
    }

    df = pd.DataFrame(data2)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

    training_data_diff = df.set_index(['trade_date'], drop=True)
    print('&', training_data_diff)

    AcfPacfPlot(training_data_diff)

    model = sm.tsa.ARIMA(endog = trainingSet['close'], order=(2, 1, 0)).fit()

    history = [x for x in trainingSet['close']]
    predictions = list()
    for t in range(testSet.shape[0]):
        model1 = sm.tsa.ARIMA(history, order=(2, 1, 0))
        model_fit = model1.fit()
        yhat = model_fit.forecast()
        yhat = np.float(yhat[0])
        predictions.append(yhat)
        obs = test_set2.iloc[t, 5]
        history.append(obs)

    predictions1 = {
        'trade_date': testSet.index[:],
        'close': predictions
    }
    predictions1 = pd.DataFrame(predictions1)
    predictions1 = predictions1.set_index(['trade_date'], drop=True)
    predictions1.to_csv('./ARIMA.csv')
    plt.figure(figsize=(10, 6))
    plt.plot(testSet['close'], label='Stock Price')
    plt.plot(predictions1, label='Predicted Stock Price')
    plt.title('ARIMA: Stock Price Prediction')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()

    model2 = sm.tsa.ARIMA(endog = data['close'], order=(2, 1, 0)).fit()
    residuals = pd.DataFrame(model2.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()
    residuals.to_csv('./ARIMA_residuals1.csv')
    EvaluationMetrics(testSet['close'],predictions)
    AdfTest(temp)
    AdfTest(temp1)

    predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
    predictions_ARIMA_diff = predictions_ARIMA_diff[3479:]
    print('#', predictions_ARIMA_diff)
    plt.figure(figsize=(10, 6))
    plt.plot(training_data_diff, label="diff_1")
    plt.plot(predictions_ARIMA_diff, label="prediction_diff_1")
    plt.xlabel('time', fontsize=12, verticalalignment='top')
    plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
    plt.title('DiffFit')
    plt.legend()
    plt.show()