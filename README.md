# Introduction
Assignment 2 tried to represent the research paper "Attention-Based CNN-LSTM and XGBoost Hybrid Model for Stock Prediction.", whose model integrated the time-series model, the convolutional neural networks with attention mechanism, the LSTM network and XGBoost regressor in a non-linear relationship, and improved the prediction accuracy.
# Data
* Stock Target: Bank of China, 601988.SH.
* Time period: Start: 20070101, End: 20220331.
* Data Source: WIND.
# Language Required
* Python 3.9
* Tensorflow 2.8.0
# File Description
1. DataProcess.py contains the data processing functions, including the ACF test function, ACF plot function, Data split function and Normalizing function, which should be runned firstly.
2. ArimaModelling.py contains the arima modeling functions. Two files: ARIMA.csv and residuals.csv will be created and used in the later modelling process, which should be runned secondly.
3. Model.py contains the modelling functions of LSTM and XGBoost, which should be runned thirdly.
4. LSTM.py and XGBoost.py apply LSTM and XGBoost on the specific data set separately.
5. main.py is the total file that gathers all the models and applies on the given data set.
# Conclusion
1. Data Description
* The stock price time series is shown as:
![ClosePrice](Figures/ClosePrice.png)
* Applying the first diff and second diff:
![FirstOrderDiff](Figures/FirstOrderDiff.png)
![SecondOrderDiff](Figures/SecondOrderDiff.png)
2. Time Series Modelling
* The ACF plot
![ACF](Figures/ACF.png)
* The Arima Stock Price Prediction
![Arima](Figures/ARIMAStockPrediction.png)
3. LSTM Modelling
* LSTM Prediction
![LSTM](Figures/LSTM.png)
* BiLSTM Prediction
![LSTM](Figures/BiLSTM.png)
4. ARIMA + XGBoost
![XGBoost](Figures/ARIMA+XGBoost.png)
5. With strong ability of modeling nonlinear model ability, the hybrid model can combine the strength of time series model and neural networks, thus better predicting the stock price.
# Reference
Shi Z, Hu Y, Mo G, et al. Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction[J]. arXiv preprint arXiv:2204.02623, 2022.
