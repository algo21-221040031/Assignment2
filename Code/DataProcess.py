import numpy as np
import pandas as pd
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 在做ARIMA前，需要先对时间序列的平稳性进行检验，使用ADF-Test
def AdfTest(temp):
    ## 做ADF检验
    adfTest = adfuller(temp)
    ## 展示结果
    adfTestResult = pd.DataFrame(index = ['Test Static Value',
                                          'p-value',
                                          'Lags Used',
                                          'Number of Observations Used',
                                          'Critical Value(1%)',
                                          'Critical Value(5%)',
                                          'Critical Value(10%)'],
                                 columns = ['Value'])
    adfTestResult['Value']['Test Static Value'] = adfTest[0]
    adfTestResult['Value']['p-value'] = adfTest[1]
    adfTestResult['Value']['Lags Used'] = adfTest[2]
    adfTestResult['Value']['Number of Observations Used'] = adfTest[3]
    adfTestResult['Value']['Critical Value(1%)'] = adfTest[4]['1%']
    adfTestResult['Value']['Critical Value(5%)'] = adfTest[4]['5%']
    adfTestResult['Value']['Critical Value(1%)'] = adfTest[4]['10%']
    print(adfTestResult)

# 对自相关函数作图
def AcfPacfPlot(seq, acf_lags = 20, pac_flags = 20):
    fig = plt.figure(figsize = (12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(seq, lags = acf_lags, ax = ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_acf(seq, lags=pac_flags, ax=ax2)
    plt.show()

# 选择最优模型的BIC
def OrderSelectIC(trainingDataDiff):
    (p,q) = sm.tsa.arma_order_select_ic(trainingDataDiff,
                                        max_ar = 6,
                                        max_ma = 4,
                                        ic = 'BIC')['bic_min_order']
    print(p,q)

def OrderSelectSearch(trainingSet):
    ## 做1阶差分
    df = trainingSet['close'].diff(1).dropna()
    pmax = 5
    qmax = 5
    bicMatrix = []
    print("^", pmax, "^^", qmax)
    for p in range(pmax + 1):
        temp = []
        for q in range(qmax + 1):
            try:
                temp.append(sm.tsa.ARIMA(trainingSet['close'],order = (p,1,q)).fit().bic)
            except:
                temp.append(None)
        bicMatrix.append(temp)
    bicMatrix = pd.DataFrame(bicMatrix)
    p,q = bicMatrix.stack().astype('float64').idxmin()
    print('p and q: %s, %s' % (p,q))

# 构建测试集和数据集
def CreateDataset(dataset, look_back = 20):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back),:]
        dataX.append(a)
        b = dataset[i + look_back,:]
        dataY.append(b)
    trainX = np.array(dataX)
    trainY = np.array(dataY)

    return trainX, trainY

# 构建模型评价指标，包括MSE, RMSE, MAE和R2
def EvaluationMetrics(yTest, yHat):
    MSE = metrics.mean_squared_error(yTest,yHat)
    RMSE = MSE ** 0.5
    MAE = metrics.mean_absolute_error(yTest, yHat)
    R2 = metrics.r2_score(yTest,yHat)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5F' % R2)

# 计算MAPE
def GetMAPE(yTest, yHat):
    sum = np.mean(np.abs((yHat - yTest) / yTest)) * 100
    return sum

def GetMAPEOrder(yTest, yHat):
    zeroIndex = np.where(yTest == 0)
    yHat = np.delete(yHat, zeroIndex[0])
    yTest = np.delete(yTest, zeroIndex[0])
    sum = np.mean(np.abs((yHat - yTest) / yTest)) * 100
    return sum

# 对数据做标准化
def NormalizeMult(data):
    data = np.array(data)
    normalize = np.arange(2 * data.shape[1], dtype = 'float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0, data.shape[1]):
        list = data[:, i]
        listlow, listhigh = np.percentile(list, [0,100])
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j,i] = (data[j,i] - listlow)/delta
    return data, normalize

def FNormalizeMult(data, normalize):
    data = np.array(data)
    listlow = normalize[0]
    listhigh = normalize[1]
    delta = listhigh - listlow
    if delta != 0:
        for i in range(len(data)):
            data[i,0] = data[i,0] * delta + listlow
    return data

def NormalizeMultUseData(data,normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    return  data

def DataSplit(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def SeriesToSupervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def PrepareData(series, n_test, n_in, n_out):
    values = series.values
    supervised_data = SeriesToSupervised(values, n_in, n_out)
    print('supervised_data', supervised_data)
    train, test = supervised_data.loc[:3499, :], supervised_data.loc[3500:, :]
    return train, test