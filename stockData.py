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

# ####################################### -INPUT HERE- #################################################################

# define period: common options would include ‘1d’ (daily), ‘1mo’ (monthly), ‘1y’ (yearly)
timeSteps = "1d"

# define start: the date to start gathering the data. For example ‘2010–1–1’. 2005-11-10 is 3600
startDate = '1996-05-1'

# define end: the date to end gathering the data. For example ‘2020–1–25’
endDate = '2020-03-3'

# define the ticker symbol
tickerSymbols = ["ADBE", "T", "MSFT", "ORCL", "IBM", "HPQ", "GE", "ADSK", "CTXS", "INTU"]
tickerFeatures = ["Open", "High", "Low", "Close", "Volume"]
tickerNames = {"ADBE": "Adobe Inc.",
               "T": "AT&T, Inc.",
               "MSFT": "Microsoft Corp.",
               "ORCL": "Oracle Corp.",
               "IBM": "International Business Machines Corp.",
               "HPQ": "Hewlett-Packard Co.",
               "GE": "General Electric",
               "ADSK": "Autodesk, Inc.",
               "CTXS": "Citrix Systems, Inc.",
               "INTU": "Intuit Inc."}

# Plot data or not
plotdata = 0

# ########################################  -DATA DOWNLOAD- ############################################################

# Download data. This is taken from https://pypi.org/project/yfinance/
stockData = yf.download(  # or pdr.get_data_yahoo(...
    # tickers list or string as well
    tickers=tickerSymbols,

    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    # period="ytd"

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
        plt.title("".join(["Stock price for ", tickerNames[symbol]]))
        plt.legend(["Open", "Close", "High", "Low"])
        plt.show()

        stockData[symbol, 'Volume'].plot()
        plt.title("".join(["Volume of stocks for ", tickerNames[symbol]]))
        plt.show()

# See your data. Output is  [Open        High         Low       Close    Volume]
#print(stockData)

# Convert data to np.array, and then to the right format
np_stockData = np.array(stockData)

# Primary numbers
N_TIMESTEPS = np_stockData.shape[0]                                                     # = 6000
print(N_TIMESTEPS)
N_COMPANIES = len(tickerSymbols)                                                        # = 10
N_FEATURES = int(np_stockData.shape[1]/N_COMPANIES)                                     # = 5
N_TIMESTEPS_PER_BATCH = 300                                                             # anything 6000 can be evenly divided with
N_BATCHES = int(np.floor(N_TIMESTEPS/N_TIMESTEPS_PER_BATCH))                            # = 20 if N_TIMESTEPS_PER_BATCH = 300

# Check for NAN-values
containing_nan_values = False
for company in tickerSymbols:
    containing_nan_values = False
    for feature in tickerFeatures:
        test_array = np.array(stockData[company, feature])
        for data in test_array:
            if not data or np.isnan(data):
                containing_nan_values = True

    if containing_nan_values:
        print("Company ", tickerNames[company], " contains NAN-values")

stockDataFinal = np.zeros([N_BATCHES, N_TIMESTEPS_PER_BATCH, N_FEATURES*N_COMPANIES])

feature_counter = 0
for feature in tickerFeatures:
    company_counter = 0
    for company in tickerSymbols:
        my_data = np.array(stockData[company, feature])
        for current_batch in range(0, N_BATCHES):
            timeSlot = N_TIMESTEPS_PER_BATCH*current_batch
            stockDataFinal[current_batch, :, feature_counter*N_COMPANIES+company_counter] = my_data[timeSlot:timeSlot+N_TIMESTEPS_PER_BATCH]
        company_counter += 1
    feature_counter += 1

# ##################################### -OUTPUT HERE- ##################################################################

# This is the output np.array, in the form [BATCH_SIZE, N_TIMESTEPS_PER_BATCH, N_FEATURES*N_COMPANIES] = [9, 150, 5*10]

# print(stockDataFinal)

# Chose output feature between 1-5 (1 is the first feature which is opening price)
my_feature = 1

# Chose company 1-10 in the list of ticker symbols above
my_company = 1

# Chose time interval (the batch) between 1 and N_BATCHES. Batch N_BATCHES is the most recent N_TIMESTEPS_PER_BATCH data points.
my_batch = 20

# Print the part of the data chosen
# print(stockDataFinal[my_batch-1, :, (my_feature-1)*N_COMPANIES+my_company-1])

# Check data for nan
for batch in stockDataFinal:
    for timestep in batch:
        for data in timestep:
            if np.isnan(data):
                print("DATA CONTAINS NAN VALUES")

print(stockDataFinal[my_batch-1, :, (my_feature-1)*N_COMPANIES+my_company-1])

# Now, just save and export stockDataFinal

# ----------------------------------------------------------------------------------------------------------------------
#  Other stuff from https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# get data on this ticker
# tickerData = yf.Ticker(tickerSymbols)
# get the historical prices for this ticker
# tickerDf = tickerData.history(period='1y', start=startDate, end=endDate)

# info on the company
# tickerInfo = tickerData.info

# get recommendation data for ticker
# tickerRec = tickerData.recommendations