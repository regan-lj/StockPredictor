import pandas as pd
from buysell import prediction
import matplotlib.pyplot as plt
# from stockData import stockDataFinal,my_feature,my_company,my_batch,N_COMPANIES
from buysell import actual_results_MAIN

data = pd.DataFrame({'Price': actual_results_MAIN})
dataSeries = data['Price']
SMA = dataSeries.rolling(window=20).mean()

# Initialize DF Indicators
data_normalize = data
df_indicators = pd.DataFrame(index=data_normalize.index)
rolling_days = 20

# Price / SMA Indicator
df_indicators['Price'] = data_normalize
df_indicators['SMA'] = data_normalize.rolling(window=rolling_days, min_periods=rolling_days).mean()
df_indicators['Price_SMA'] = df_indicators['Price'] / df_indicators['SMA']

# Bollinger Bands Indicator
std = data_normalize.rolling(window=rolling_days, min_periods=rolling_days).std()
df_indicators['Upper_band'] = df_indicators['SMA'] + (2 * std['Price'])
df_indicators['Lower_band'] = df_indicators['SMA'] - (2 * std['Price'])
df_indicators['BB_value'] = (df_indicators['Price'] - df_indicators['Lower_band']) / (df_indicators['Upper_band'] - df_indicators['Lower_band'])

# Momentum Indicator
df_indicators['Momentum'] = df_indicators['Price'] / data_normalize.shift(rolling_days)['Price'] - 1

price_sma = df_indicators['Price_SMA']
bbv = df_indicators['BB_value']
momentum = df_indicators['Momentum']

day_list = []
order_list = []
price_list = []
status = "SELL"
for day in range(len(df_indicators)):
    if (bbv.iloc[day] < 0.0) and (momentum.iloc[day] < 0.0) and (price_sma.iloc[day] < 0.95) and (status == "SELL"):
        status = "BUY"
        day_list.append(day)
        order_list.append("BUY")
        price_list.append(df_indicators['Price'].iloc[day])
    elif (bbv.iloc[day] > 1.0) and (momentum.iloc[day] > 0.0) and (price_sma.iloc[day] > 1.05) and (status == "BUY"):
        status = "SELL"
        day_list.append(day)
        order_list.append("SELL")
        price_list.append(df_indicators['Price'].iloc[day])

# print("------------ OUTPUT ------------")
# print(day_list, order_list, price_list)
# print("------------ OUTPUT ------------")
orders = pd.DataFrame({'Day': day_list, 'Order': order_list, 'Price': price_list})
# print(orders)

# plt.figure(4)
# plt.style.use('seaborn-whitegrid')
# df_indicators[['Momentum', 'Price_SMA', 'BB_value']].plot(figsize=(20, 7))
# plt.axhline(y=1.05, linestyle='--', c='b')
# plt.axhline(y=0.95, linestyle='--', c='b')
# plt.axhline(y=1, linestyle='--', c='g')
# plt.axhline(y=.1, linestyle='-.', c='g')
# plt.axhline(y=0, linestyle='-.', c='r')
# plt.grid(axis='both')
# plt.title("Stock Data Indicators")
# plt.ylabel("Value")
# plt.xlabel("Date Index")
# plt.legend(frameon=True, loc=0, ncol=1, fontsize=10, borderpad=.6)
# plt.xlim([df_indicators.index.min()+rolling_days, df_indicators.index.max()])
# plt.show()

# plt.figure(1)
# df_indicators[['Price', 'SMA', 'Price_SMA']].plot(figsize=(20,7))
# plt.axhline(y=1.05, linestyle='--',c='g')
# plt.axhline(y=0.95, linestyle='--',c='r')
# plt.grid(axis='both')
# plt.show()
#
# plt.figure(2)
# df_indicators[['BB_value', 'Upper_band', 'Lower_band', 'SMA']].plot(figsize=(20, 7))
# plt.axhline(y=1, linestyle='--', c='g')
# plt.axhline(y=0, linestyle='--', c='r')
# plt.grid(axis='both')
# plt.show()
#
# plt.figure(3)
# df_indicators[['Momentum']].plot(figsize=(20, 7))
# plt.axhline(y=.1, linestyle='-.', c='g')
# plt.axhline(y=0, linestyle='-.', c='r')
# plt.grid(axis='both')
# plt.show()