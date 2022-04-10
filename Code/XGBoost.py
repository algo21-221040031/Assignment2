import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataProcess import *
from Model import WalkForwardValidation

if __name__ == "__main__":
    
    data = pd.read_csv('../Data/601988.SH.csv')
    data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
    # 取出close收盘价列
    close = data.pop('close')
    data.insert(5, 'close', close)

    data1 = data.iloc[3501:, 5]
    residuals = pd.read_csv('./ARIMA_residuals1.csv')
    residuals.index = pd.to_datetime(residuals['trade_date'])
    residuals.pop('trade_date')
    merge_data = pd.merge(data, residuals, on='trade_date')
    #merge_data = merge_data.drop(labels='2007-01-04', axis=0)
    time = pd.Series(data.index[3501:])

    Lt = pd.read_csv('./ARIMA.csv')
    Lt = Lt.drop('trade_date', axis=1)
    Lt = np.array(Lt)
    Lt = Lt.flatten().tolist()

    train, test = PrepareData(merge_data, n_test=180, n_in=6, n_out=1)

    y, yhat = WalkForwardValidation(train, test)
    plt.figure(figsize=(10, 6))
    plt.plot(time, y[1:], label='Residuals')
    plt.plot(time, yhat[1:], label='Predicted Residuals')
    plt.title('ARIMA+XGBoost: Residuals Prediction')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()

    finalpredicted_stock_price = [i + j for i, j in zip(Lt, yhat)]
    #print('final', finalpredicted_stock_price)
    EvaluationMetrics(data1, finalpredicted_stock_price)
    plt.figure(figsize=(10, 6))
    plt.plot(time, data1, label='Stock Price')
    plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
    plt.title('ARIMA+XGBoost: Stock Price Prediction')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()
