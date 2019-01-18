# Air Passengers ARIMA model 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from pyramid.arima import auto_arima


# Load the Dataset
data = pd.read_csv("AirPassengers.csv")

# Convert the month variable in proper date format
data["Month"] = pd.to_datetime(data["Month"])
data.head()

# Set the month as index
data.set_index("Month",inplace=True)
data= pd.Series(data["#Passengers"])
# Plot the data
plt.plot(data)

# Create the stationary function
from statsmodels.tsa.stattools import adfuller
def stationary_test(timeseries):
    rolling_mean = timeseries.rolling(12).mean()
    rolling_std = timeseries.rolling(12).std()
    
    # plot the rolling statistics
    orig = plt.plot(timeseries, color = "blue", label = "original")
    mean = plt.plot(rolling_mean, color = "red", label = "Mean")
    std = plt.plot(rolling_std, color = "black", label = "Std")
    
    # Perform Dickey - fuller test
    print("Result of Dickey-Fyller test")
    DF_test = adfuller(timeseries,autolag='AIC')
    print("ADF statistics :", DF_test[0])
    print("p-value :", DF_test[1])
    print("No. of Lags used :", DF_test[2])
    print("No. of observation used :", DF_test[3])
    print("Critical values :")
    for key, value in DF_test[4].items():
        print((key, value))
    if DF_test[1] <0.05:
        print("We fail to accept the null hypothesis")
        print("Data is stationary")
    else:
        print("We rejected the alternate hypothesis")
        print("Data is non stationary")
        
stationary_test(data)    
 
# Make the stationary data by removing trend only
data_log = np.log(data)
stationary_test(data_log)

# 1) Difference between series log and moving avg
moving_avg = data_log.rolling(12).mean()
data_diff_log = data_log-moving_avg
data_diff_log.dropna(inplace=True)
stationary_test(data_diff_log)
    
# 2) Difference between log series and exp weighted avg
ewma = pd.Series.ewm
expweighted_avg = ewma(data_log, span=12).mean()
diff_data_log_ewma = data_log - expweighted_avg
diff_data_log_ewma.head()
stationary_test(diff_data_log_ewma)
    
# Make the stationary data by removing Trend and Seasonality
    
# 1) Differencing
data_log_diff = data_log - data_log.shift()
data_log_diff.dropna(inplace = True)
stationary_test(data_log_diff)

data_log_diff_1  = data_log_diff - data_log_diff.shift()
data_log_diff_1.dropna(inplace = True)
stationary_test(data_log_diff_1)
    
# 2) Decomposing
decomposition = seasonal_decompose(data_log)
decomposition.plot()
"""trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
    
plt.subplot(411)
plt.plot(data_log, label = "original")
plt.legend(loc = "best")
plt.subplot(412)
plt.plot(trend, label = "trend")
plt.legend(loc = "best")
plt.subplot(413)
plt.plot(seasonal, label = "seasonal")
plt.legend(loc = "best")
plt.subplot(414)
plt.plot(residual, legend = "Residual")
plt.legend(loc = "best")
plt.tight_layout()
    
# We check the stationarity on residual model
residual.dropna(inplace = True)
stationary_test(residual)"""


# Best model with Auto Arima
result = auto_arima(data, start_p=2,
                    d=1, start_q=2,
                    max_p=5, max_d = 5,
                    max_q=5, start_P=0,
                    D=1, start_Q=1,
                    max_P=5, max_D=5,
                    max_Q=5, seasonal=True,
                    stationary =False,
                    m=12,   
                    trace=True,
                    suppress_warnings=True,
                    stepwise=True,
                    information_criterion= "aic",
                    error_action="ignore"
                    )

result.aic()
result.summary()

# Split the data into Training and Testing dataset
train = data.loc["1949-01-01":"1958-12-31"]
test = data.loc["1959-01-01":]

# Fit the model into train dataset
result.fit()
prediction = result.predict(n_periods=24)
mse = ((prediction - test) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
prediction = pd.DataFrame(prediction, index = test.index, columns = ["Pred"])
prediction.head()
prediction.tail()
# Concate the prediction and test dataset
pd.concat([test, prediction], axis = 1).plot()
pd.concat([data,prediction], axis=1).plot()



# Find the p,d,q, P, D, Q parameters of non seasonal and  Seasonal data resp
p=d=q = range(0,3)
Non_seasonal_pdq = list(itertools.product(p, d, q))
Seasonal_PDQ = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Grid search to find the optimal parameters with lowest AIC
warnings.filterwarnings("ignore")
best_result = [0, 0, 10000000]
for param in Non_seasonal_pdq:
    for param_seasonal in Seasonal_PDQ:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            print('ARIMA{} x {} - AIC: {}'.format(param, param_seasonal, results.aic))

            if results.aic < best_result[2]:
                best_result = [param, param_seasonal, results.aic]
        except:
            continue
            
print('\nBest Result:', best_result)
#best_result = [(2,1,2), (0, 2, 2, 12)]

# Best Model with optimal paramters 
mod = sm.tsa.statespace.SARIMAX(data,
                                order=(best_result[0][0], best_result[0][1], best_result[0][1]),
                                seasonal_order=(best_result[1][0], best_result[1][1], best_result[1][2], best_result[1][3]),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
prediction = results.predict(n_periods=24)
mse = ((prediction - test) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
prediction = pd.DataFrame(prediction, index = test.index, columns = ["Pred"])
prediction.head()
prediction.tail()
# Concate the prediction and test dataset
pd.concat([test, prediction], axis = 1).plot()
pd.concat([data,prediction], axis=1).plot()
    
