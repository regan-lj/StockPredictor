# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 03:41:42 2020

@author: sagar

I created a Python script that is used for evaluation of our Machine Learning Stock Predictor.

In order for a python script to stimulate buying and selling stocks, we need to give the program buy or sell signals. These signals come from different indicators, such as simple moving average, volume, and more.

The most basic signal is a simple moving average. In this program, I use a 50 day moving average to give this program buy and sell signals. I start off with a principle of $1000, and stimulate the market for 5 years of test data. 

This is no where close to being perfect, but gives us a baseline idea of how our stocks should perform in the given time period. 

"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

today = date.today() # Get todays date in a datetime format
start_date = today - timedelta(days=5*365) # Get the date from 5 years ago in a datetime format

# Use pandas_reader.data.DataReader to load the desired data for specified stock from yahoo starting from the start date to today
panel_data = data.DataReader('MSFT', 'yahoo', start_date)

#print(panel_data) # Returns Open price, Close price, High price, Low price, and Volume of MSFT stock 

msft = panel_data['Close'] # Use only the closing price of MSFT

# Calculate the 10, 50, and 100 days moving averages of the closing prices
short_rolling_msft = msft.rolling(window=10).mean()
med_rolling_msft = msft.rolling(window=50).mean()
long_rolling_msft = msft.rolling(window=100).mean()

med_rolling_msft = med_rolling_msft.dropna() # Since the rolling average is not calculated for the first few days, the result is a NA value, so I drop those values
diff = len(msft) - len(med_rolling_msft) # Get the number of dropped data points
msft = msft[diff:] # Drop the MSFT closing price data for the values that the rolling average is not calculated

"""

The way this program generates signals is by comparing the closing price and simple moving average (in this case I am using a 50 day moving average)

When the simple moving average crosses from below the current price to above the closing price, the program generates a sell signal
When the simple moving average crosses from above the current price to below the closing price, the program generates a buy signal
When the simple moving average does not cross the current price, that means to hold the stock

"""


# False means mean is less than current value
# True means mean is greater than current value
# 0 -> 1 means sell
# 1 -> 0 means buy

hasStock = 0
money = 1000
status = med_rolling_msft[0] > msft[0] # Initial status (buy or sell) of the stock

for i in range(1, len(med_rolling_msft)): # For loop in range of each day
    newStatus = med_rolling_msft[i] > msft[i] # Compare if the rolling average or current price is greater
    if status != newStatus: # Test if the status is different, if it is different, it means either a buy or sell signal 
        print(msft.index[i]) # print the current price of the stock
        if newStatus and hasStock: # newStatus of 1 means sell, and make sure we have stocks to sell
            print("Sell %d shares at $%.2f" % (hasStock, msft[i]))
            money += hasStock * msft[i]
            hasStock = 0
        elif (~newStatus and ~hasStock): # newStatus of 0 means buy, and should not have any stocks before making the buy
            hasStock = money // msft[i] # calculate the max amount of stocks that we can buy 
            print("Buy %d shares at $%.2f" % (hasStock, msft[i]))
            money -= hasStock * msft[i]
        print("Money: $%.2f, Shares: %d, Value: $%.2f" % (money, hasStock, (money + hasStock*msft[i])))
        print()
    status = newStatus # update the status
    
# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(msft.index, msft, label='MSFT')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='10 days rolling')
ax.plot(med_rolling_msft.index, med_rolling_msft, label='50 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()