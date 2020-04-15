import pandas
import numpy
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from buysell import prediction
from benchMark import orders as benchmarkOrders
from buysell import orders as mlOrders # Machine Learning Buy Sell Output

print("------------ BENCHMARK ORDERS ------------")
print(benchmarkOrders)
print("------------ ML ORDERS ------------")
print(mlOrders)

principal = 10000

portfolioValues = pd.DataFrame({'Price': prediction})
print(portfolioValues)

benchmarkShares = 0
benchmarkCash = principal
benchmarkPortfolioValue = []
benchmarkCount = 0

MLShares = 0
MLCash = principal
MLPortfolioValue = []
MLCount = 0
for i in range(len(portfolioValues)):
    if i in benchmarkOrders['Day'].values:
        order = benchmarkOrders['Order'].iloc[benchmarkCount]
        day = benchmarkOrders['Day'].iloc[benchmarkCount]
        price = benchmarkOrders['Price'].iloc[benchmarkCount]
        print(day, order, price)
        benchmarkCount += 1
        if order == "BUY":
            benchmarkShares = benchmarkCash // price
            benchmarkCash -= benchmarkShares * price
        elif order == "SELL":
            benchmarkCash += benchmarkShares * price
            benchmarkShares = 0

    value = benchmarkShares * portfolioValues['Price'].iloc[i] + benchmarkCash
    benchmarkPortfolioValue.append(value)

    if i in mlOrders['Day'].values:
        order = mlOrders['Order'].iloc[MLCount]
        day = mlOrders['Day'].iloc[MLCount]
        price = mlOrders['Price'].iloc[MLCount]
        print(day, order, price)
        MLCount += 1
        if order == "BUY":
            MLShares = MLCash // price
            MLCash -= MLShares * price
        elif order == "SELL":
            MLCash += MLShares * price
            MLShares = 0

    value = MLShares * portfolioValues['Price'].iloc[i] + MLCash
    MLPortfolioValue.append(value)

portfolioValues['Benchmark'] = benchmarkPortfolioValue
portfolioValues['ML'] = MLPortfolioValue
print(portfolioValues)

price_norm = portfolioValues['Price'] / portfolioValues['Price'].iloc[0]
benchmark_norm = portfolioValues['Benchmark'] / portfolioValues['Benchmark'].iloc[0]
ml_norm = portfolioValues['ML'] / portfolioValues['ML'].iloc[0]

# plt.figure()
# plt.style.use('seaborn-whitegrid')
# plt.plot(price_norm, label="Stock Price")
# plt.plot(benchmark_norm, label="Benchmark")
# plt.plot(ml_norm, label="ML Algorithm")
# plt.legend(frameon=True, loc=0, ncol=1, fontsize=10, borderpad=.6)
# plt.title("Portfolio Growth (Stock vs Benchmark vs ML Algorithm")
# plt.ylabel("Normalized Portfolio Value")
# plt.xlabel("Date Index")
# plt.xlim([portfolioValues.index.min(), portfolioValues.index.max()])
# plt.show()

buysML = mlOrders[mlOrders['Order'] == 'BUY']
sellsML = mlOrders[mlOrders['Order'] == 'SELL']

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.plot(portfolioValues['Price'], label='Stock Price')
plt.scatter(buysML.Day, buysML.Price, c='#4bd81d', label='Buy Point')
plt.scatter(sellsML.Day, sellsML.Price, c='#ff001e', label='Sell Point')
plt.legend(frameon=True, loc=0, ncol=1, fontsize=10, borderpad=.6)
plt.title('Machine Learning Buy/Sell Points', fontSize=15)
plt.ylabel('Stock Price', fontSize=12)
plt.xlabel('Date Index', fontSize=12)
plt.xlim([portfolioValues.index.min(), portfolioValues.index.max()])
plt.show()

buysBM = benchmarkOrders[benchmarkOrders['Order'] == 'BUY']
sellsBM = benchmarkOrders[benchmarkOrders['Order'] == 'SELL']

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.plot(portfolioValues['Price'], label='Stock Price')
plt.scatter(buysBM.Day, buysBM.Price, c='#4bd81d', label='Buy Point')
plt.scatter(sellsBM.Day, sellsBM.Price, c='#ff001e', label='Sell Point')
plt.legend(frameon=True, loc=0, ncol=1, fontsize=10, borderpad=.6)
plt.title('Benchmark Buy/Sell Points', fontSize=15)
plt.ylabel('Stock Price', fontSize=12)
plt.xlabel('Date Index', fontSize=12)
plt.xlim([portfolioValues.index.min(), portfolioValues.index.max()])
plt.show()

"""
AFTER THIS POINT

EVALUATION.py
TAKE BUY/SELL DATAFRAME and PRINCIPLE AMOUNT
CALCULATE PORTFOLIO VALUE OVER TIME
GRAPH BENCHMARK 1.0 (SMA), BENCHMARK 2.0 (INDICATORS), STOCK PRICE, and MACHINE LEARNING OUTCOME (all normalized)

"""
