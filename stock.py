# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 03:41:42 2020

@author: sagar
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

today = date.today()
start_date = today - timedelta(days=5*365)

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader('MSFT', 'yahoo', start_date)

#print(panel_data)

#print(start_date)

# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
#tickers = ['AAPL', 'MSFT', 'GOOG']

# We would like all available data from last 5 years
#start_date = '2015-01-01'
#end_date = '2020-03-06'

msft = panel_data['Close']

# Calculate the 20, 40, and 100 days moving averages of the closing prices
short_rolling_msft = msft.rolling(window=20).mean()
med_rolling_msft = msft.rolling(window=100).mean()
long_rolling_msft = msft.rolling(window=200).mean()

med_rolling_msft = med_rolling_msft.dropna()
diff = len(msft) - len(med_rolling_msft)
msft = msft[diff:]

# False means mean is less than current value
# True means mean is greater than current value
# 0 -> 1 means sell
# 1 -> 0 means buy

hasStock = 0
money = 1000
status = med_rolling_msft[0] > msft[0]

for i in range(1, len(med_rolling_msft)):
    newStatus = med_rolling_msft[i] > msft[i]
    if status != newStatus:
        print(msft.index[i])
#        print("New Status: %d, Has Stock: %d" % (newStatus, hasStock))
        if newStatus and hasStock:
            print("Sell %d shares at $%.2f" % (hasStock, msft[i]))
            money += hasStock * msft[i]
            hasStock = 0
        elif (~newStatus and ~hasStock):
            hasStock = money // msft[i]
            print("Buy %d shares at $%.2f" % (hasStock, msft[i]))
            money -= hasStock * msft[i]
        print("Money: $%.2f, Shares: %d, Value: $%.2f" % (money, hasStock, (money + hasStock*msft[i])))
        print()
    status = newStatus
    
# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(msft.index, msft, label='MSFT')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
ax.plot(med_rolling_msft.index, med_rolling_msft, label='100 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='200 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()