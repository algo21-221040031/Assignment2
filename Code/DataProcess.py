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

def AcfPacfPlot(seq, acf_lags = 20, pac_flags = 20):
    fig = plt.figure(figsize = (12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(seq, lags = acf_lags, ax = ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_acf(seq, lags=pac_flags, ax=ax2)
    plt.show()




if __name__ == "main":
    df = pd.read_csv("./Data/600519.SH.csv")
    df.head()
