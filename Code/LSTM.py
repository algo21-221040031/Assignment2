from random import triangular
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import adam_v2
# from tensorflow.keras.optimizers import Adam
from numpy.random import seed
from DataProcess import *
from Model import lstm

if __name__ == "__main__":

    # GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices([gpus[0]], "GPU")

    seed(1)
    tf.random.set_seed(1)

    n_timestamp = 10
    n_epochs = 50
    # ====================================
    #      model type：
    #            1. single-layer LSTM
    #            2. multi-layer LSTM
    #            3. bidirectional LSTM
    # ====================================
    model_type = 3

    # 输入原数据
    originalData = pd.read_csv('../Data/601988.SH.csv')
    originalData.index = pd.to_datetime(originalData['trade_date'], format = '%Y%m%d')
    originalData = originalData.loc[:,['open','high','low','close','amount']]

    # 输入ARIMA模型的残差项
    residualData = pd.read_csv('./ARIMA_residuals1.csv')
    residualData.index = pd.to_datetime(residualData['trade_date'])
    residualData = residualData.drop('trade_date', axis = 1)

    # 输入ARIMA模型结果
    Lt = pd.read_csv('./ARIMA.csv')
    idx = 3500
    trainingSet = residualData.iloc[1:idx, :]
    testSet = residualData.iloc[idx:, :]
    originalTrainingSet = originalData.iloc[1:idx, :]
    originalTestSet = originalData.iloc[idx:,:]

    # 数据做归一化处理
    Scaler = MinMaxScaler(feature_range = (0,1))
    originalScaler = MinMaxScaler(feature_range = (0,1))
    scaledTrainingSet = Scaler.fit_transform(trainingSet)
    scaledTestSet = Scaler.fit_transform(testSet)
    scaledOriginalTrainingSet = originalScaler.fit_transform(originalTrainingSet)
    scaledOriginalTestSet = originalScaler.fit_transform(originalTestSet)

    # 数据分割，方便后续建模
    ## 自相关模型，因变量和自变量均为自身
    xTrain, yTrain = DataSplit(scaledTrainingSet, n_timestamp)
    originalXTrain, originalYTrain = DataSplit(scaledOriginalTrainingSet, n_timestamp)
    xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], 1)
    originalXTrain = originalXTrain.reshape(
        originalXTrain.shape[0], originalXTrain.shape[1], 5)
    
    xTest, yTest = DataSplit(scaledTestSet, n_timestamp)
    originalXTest, originalYTest = DataSplit(
        scaledOriginalTestSet, n_timestamp)
    xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 1)
    originalXTest = originalXTest.reshape(
        originalXTest.shape[0], originalXTest.shape[1], 5)
    
    # 使用LSTM对数据建模
    Model, originalModel = lstm(model_type, xTrain, originalXTrain)
    print(Model.summary())

    adam = adam_v2.Adam(learning_rate=0.01)
    Model.compile(optimizer=adam,
              loss='mse')
    originalModel.compile(optimizer=adam,
                   loss='mse')
    
    History = Model.fit(xTrain, yTrain,
                        batch_size=32,
                        epochs=n_epochs,
                        validation_data=(xTest, yTest),
                        validation_freq=1)
    originalHistory = originalModel.fit(originalXTrain, originalYTrain,
                                batch_size=32,
                                epochs=n_epochs,
                                validation_data=(originalXTest, originalYTest),
                                validation_freq=1)


    plt.figure(figsize=(10, 6))
    plt.plot(History.history['loss'], label='Training Loss')
    plt.plot(History.history['val_loss'], label='Validation Loss')
    plt.title('residuals: Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(originalHistory.history['loss'], label='Training Loss')
    plt.plot(originalHistory.history['val_loss'], label='Validation Loss')
    plt.title('LSTM: Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # 基于模型结果做预测
    originalPredictedStockPrice = originalModel.predict(originalXTest)
    originalPredictedStockPrice = originalScaler.inverse_transform(
        originalPredictedStockPrice)
    originalPredictedStockPriceList = np.array(
        originalPredictedStockPrice[:, 3]).flatten().tolist()
    originalPredictedStockPriceFinal = {
        'trade_date': originalData.index[idx+10:],
        'close': originalPredictedStockPriceList
    }
    originalPredictedStockPriceFinal = pd.DataFrame(originalPredictedStockPriceFinal)
    originalPredictedStockPriceFinal = originalPredictedStockPriceFinal.set_index(
        ['trade_date'], drop=True)
    originalRealStockPrice = originalScaler.inverse_transform(originalYTest)
    originalRealStockPriceList = np.array(
        originalRealStockPrice[:, 3]).flatten().tolist()
    originalRealStockPrice1 = {
        'trade_date': originalData.index[idx+10:],
        'close': originalRealStockPriceList
    }
    originalRealStockPrice1 = pd.DataFrame(originalRealStockPrice1)
    originalRealStockPrice1 = originalRealStockPrice1.set_index(
        ['trade_date'], drop=True)

    predictedStockPrice = Model.predict(xTest)
    predictedStockPrice = Scaler.inverse_transform(predictedStockPrice)
    predictedStockPriceList = np.array(
        predictedStockPrice[:, 0]).flatten().tolist()

    predictedStockPrice1 = {
        'trade_date': residualData.index[idx+10:],
        'close': predictedStockPriceList
    }
    predictedStockPrice1 = pd.DataFrame(predictedStockPrice1)

    predictedStockPrice1 = predictedStockPrice1.set_index(
        ['trade_date'], drop=True)

    realStockPrice = Scaler.inverse_transform(yTest)
    finalPredictedStockPrice = pd.concat([Lt, predictedStockPrice1]).groupby(
        'trade_date')['close'].sum().reset_index()
    finalPredictedStockPrice.index = pd.to_datetime(
        finalPredictedStockPrice['trade_date'])  # 将时间格式改变一下
    finalpredictedStockPrice = finalPredictedStockPrice.drop(
        ['trade_date'], axis=1)

    plt.figure(figsize=(10, 6))
    # print('yuan_real', yuan_real_stock_price1)
    plt.plot(originalData.loc['2021-06-22':, 'close'], label='Stock Price')
    plt.plot(finalPredictedStockPrice['close'], label='Predicted Stock Price')
    plt.title('BiLSTM: Stock Price Prediction')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(originalRealStockPrice1['close'], label='Stock Price')
    plt.plot(originalPredictedStockPriceFinal['close'], label='Predicted Stock Price')
    plt.title('LSTM: Stock Price Prediction')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()

    yHat = originalData.loc['2021-06-22':, 'close']
    EvaluationMetrics(finalPredictedStockPrice['close'], yHat)