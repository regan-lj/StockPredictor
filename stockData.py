# The key arguments here are:
#   period: the frequency at which to gather the data;
#   # common options would include ‘1d’ (daily), ‘1mo’ (monthly), ‘1y’ (yearly)
#   start: the date to start gathering the data. For example ‘2010–1–1’
#   end: the date to end gathering the data. For example ‘2020–1–25’
# Your result should be a Pandas dataframe containing daily historical stock price data for Microsoft.
# Key fields include:
#   Open: the stock price at the beginning of that day/month/year
#   Close: the stock price at the end of that day/month/year
#   High: the highest price the stock achieved that day/month/year
#   Low: the lowest price the stock achieved that day/month/year
#   Volume: How many shares were traded that day/month/year
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ######################################## INPUT HERE #################################################################

# define period: common options would include ‘1d’ (daily), ‘1mo’ (monthly), ‘1y’ (yearly)
timeSteps = "1mo"

# define start: the date to start gathering the data. For example ‘2010–1–1’
startDate = '2010-1-01'

# define end: the date to end gathering the data. For example ‘2020–1–25’
endDate = '2020-03-21'

# define the ticker symbol
tickerSymbols = ["SPY", "AAPL", "AMZN", "TSLA", "GOOGL", "BAC"]
tickerFeatures = ["Open", "High", "Low", "Close", "Volume"]

# Plot data or not
plotdata = 0

# #####################################################################################################################

# Download data. This is taken from https://pypi.org/project/yfinance/
stockData = yf.download(  # or pdr.get_data_yahoo(...
    # tickers list or string as well
    tickers=tickerSymbols,

    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    # period="ytd",

    start=startDate,
    end=endDate,
    period=timeSteps,

    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    # interval = '1d',

    # group by ticker (to access via data['SPY']) (optional, default is 'column')
    group_by='ticker',

    # adjust all OHLC automatically (optional, default is False)
    auto_adjust=True,

    # download pre/post regular market hours data (optional, default is False)
    prepost=True,

    # use threads for mass downloading? (True/False/Integer) (optional, default is True)
    threads=True,

    # proxy URL scheme use use when downloading? (optional, default is None)
    proxy=None
)

# PLOT DATA #
if plotdata:
    for symbol in tickerSymbols:
        stockData[symbol, 'Open'].plot()
        stockData[symbol, 'Close'].plot()
        stockData[symbol, 'High'].plot()
        stockData[symbol, 'Low'].plot()
        plt.title(['Stock price for', symbol])
        plt.legend(["Open", "Close", "High", "Low"])
        plt.show()

        stockData[symbol, 'Volume'].plot()
        plt.title(["Volume of stocks for ", symbol])
        plt.show()

# See your data. Output is  [Open        High         Low       Close     Volume]
print(stockData)

# Convert data to np.array, and then to the right format
np_stockData = np.array(stockData)

N_TIMESTEPS = np_stockData.shape[0]
N_COMPANIES = len(tickerSymbols)
N_FEATURES = int(np_stockData.shape[0]/N_COMPANIES)

np_stockData_resh = np.zeros([N_COMPANIES, N_TIMESTEPS, N_FEATURES])

icounter = 0
for company in tickerSymbols:
    kcounter = 0
    for feature in tickerFeatures:
        np_stockData_resh[icounter, :, kcounter] = np.array(stockData[company, feature])
        kcounter += 1
    icounter += 1

#  Example of how to access data for company nr 1 and feature nr 2:
print(np_stockData_resh[1, :, 2])



#  ----------- Other stuff from https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# get data on this ticker
# tickerData = yf.Ticker(tickerSymbols)
# get the historical prices for this ticker
# tickerDf = tickerData.history(period='1y', start=startDate, end=endDate)

# info on the company
# tickerInfo = tickerData.info

# get recommendation data for ticker
# tickerRec = tickerData.recommendations
