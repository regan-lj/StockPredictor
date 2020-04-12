import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from stockData import stockDataFinal as np_stockData_resh

def time_differencing(series):
    """ Rather than considering the index directly, we are calculating
        the difference between consecutive time steps.
        PARAMETERS
            series : raw stock index divided at different batches
        RETURNS
            time-differenced stock index data divided at different batches
    """
    copy_series = series.copy()
    for index in range(len(series)):
        copy_series[index][0][0] = (series[index][0][0] - series[index][0][0])
    for batch_i in range(1, len(series)):
        for time_i in range(len(series[batch_i])):
            copy_series[batch_i][time_i] = series[batch_i][time_i] - series[batch_i][time_i-1]
    ans = copy_series
    return ans


series = time_differencing(np_stockData_resh) #export this to main script

plt.clf()
plt.plot(np_stockData_resh[1, :, 2])
plt.plot(series[1, :, 2])
plt.legend(["Index", "Time-diff"])
plt.show()


#Testing part: check if data is stationary. Treshhold is that "ADF Statistic"
#shoud be below critacal values and p-value < 0.05.
#source for theory: https://machinelearningmastery.com/time-series-data-stationary-python/

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
