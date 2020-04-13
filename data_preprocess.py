import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from stockData import stockDataFinal as np_stockData_resh

def time_differencing(series):
    """ Rather than considering the index directly, we are calculating
        the difference between consecutive time steps.
        PARAMETERS
            series : raw stock index divided at different batches
        RETURNS
            time-differenced stock index data divided at different batches
    """
    batch_sz = len(series)
    time_steps = len(series[1])
    n_features = len(series[1][1])
    copy_series = series.copy()
    """
    for index in range(len(series)):
        copy_series[index][0][:] = (series[index][0][:] - series[index][0][:]) #could also remove...
    for batch_i in range(len(series)):
        for time_i in range(1, len(series[batch_i])):
            copy_series[batch_i][time_i] = series[batch_i][time_i] - series[batch_i][time_i-1]
    """
    temp = np.zeros((batch_sz, time_steps-1, n_features))
    for i in range(batch_sz):
        temp[i] = np.delete(copy_series[i], 0, 0)
    for batch_i in range(len(series)):
        for time_i in range(0, len(series[batch_i])-1):
            temp[batch_i][time_i] = series[batch_i][time_i+1] - series[batch_i][time_i]
    ans = temp
    return ans

def reverse_differencing(last_value, ML_output):  #use this function on ML preciction before sending to buysell.py
    """ Integrate data into original data format, e.g. raw stack price
        last_value is value before differncing for feature we want to predict at last time step
    """
    output = ML_output.copy()
    output[0] = ML_output[0] + last_value
    for i in range(1, len(ML_output)):
        output[i] = ML_output[i] + output[i-1]
    return output

series = time_differencing(np_stockData_resh) #export this to main script

print(len(np_stockData_resh[1, :, 2]))
print(len(series[1, :, 2]))
#print(len(final_pred))

plt.clf()
plt.plot(np_stockData_resh[1, :, 2])
plt.plot(series[1, :, 2])
#plt.plot(final_pred)
plt.legend(["Index", "Time-diff"])
plt.show()

#Testing part: check if data is stationary. Treshhold is that "ADF Statistic"
#shoud be below critacal values and p-value < 0.05.
#source for theory: https://machinelearningmastery.com/time-series-data-stationary-python/
"""
X = np_stockData_resh[1, :, 2].tolist() #raw data
X_mod = series[1, :, 2].tolist()        #processed data

result = adfuller(X)
print('X ADF Statistic: %f' % result[0])
print('X p-value: %f' % result[1])
print('X Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

result2 = adfuller(X_mod)
print('X_mod ADF Statistic: %f' % result2[0])
print('X_mod p-value: %f' % result2[1])
print('X_mod Critical Values:')
for key, value in result2[4].items():
	print('\t%s: %.3f' % (key, value))
"""
