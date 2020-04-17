"""
Export final_result: a numpy array with stock prices in the first row and the corresponding signals in the second row.
Until I have it set up for the ML, line 295 can be altered to change the stock.

Signals
----------
     1: buy
    -1: sell
     0: hold
"""

import matplotlib.pyplot as plt
import numpy as np
from stockData import stockDataFinal,my_feature,my_company,my_batch,N_COMPANIES
import pandas as pd

sensitivity = 4

def determine_maxima_minima(data, num_steps, sensitivity):
    """
    Identifies all local maximum and minimum indices.
    Filters out the insignificant maxima/minima.

    Parameters
    ----------
    data : float array
        A matrix containing the stock price at each day
    num_steps : int
        The size of data
    sensitivity : int
        How susceptible the buy/sell signals are to changes in price
        # 4 was ideal for mock data - may need to change with real data

    Returns
    -------
    signs
        An array containing signals at each day
        +1 means buy, -1 means sell, 0 means hold
    """

    # Find local minima/maxima (very specific)
    buy = np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]
    sell = np.r_[True, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], True]

    signs = np.zeros(num_steps) # Will be +1 for buy and -1 for sell

    # Map buy/sell to above vectors
    for x in range (0, num_steps):
        if buy[x] == True:
            signs[x] = 1
        if sell[x] == True:
            signs[x] = -1

    # Ensures that the last signal is to sell and that the first signal is to buy
    # NOTE: Don't need this for the dynamic version
    if signs[-1] == 1:
        signs[-1] = 0
    if signs[0] == -1:
        signs[0] = 0

    ###################### Get rid of volatile and redundant values ####################

    # If there are 3 points close together with zeros on either side -> best of edges remains
    for x in range (2,num_steps-2):
        if signs[x-2] == 0 and signs[x-1] != 0 and signs[x] != 0 and signs[x+1] != 0 and signs[x+2] == 0:
            if (signs[x] == 1 and data[x-1] > data[x+1]) or (signs[x] == -1 and data[x-1] < data[x+1]):
                signs[x+1] = 0
            else:
                signs[x-1] = 0
            signs[x] = 0

    # Get the edge points that can't be iterated over
    if signs[0] == 1 and signs[1] == -1 and signs[2] == 1:
        if signs[0] < signs[2]:
            signs[2] = 0
        else:
            signs[0] = 0
        signs[1] = 0
    if signs[-1] == -1 and signs[-2] == 1 and signs[-3] == -1:
        if signs[-1] > signs[-3]:
            signs[-3] = 0
        else:
            signs[-1] = 0
        signs[-2] = 0

    # If there are 2 points close together with zeros on either side -> make both zero
    # These are impossible to be local maxima/minima
    for x in range (1,num_steps-2):
        if signs[x-1] == 0 and signs[x] != 0 and signs[x+1] != 0 and signs[x+2] == 0:
            signs[x] = 0
            signs[x+1] = 0

    # Get the edge points that can't be iterated over
    if signs[0] != 0 and signs[1] != 0 and signs[2] == 0:
        signs[0] = 0
        signs[1] = 0
    if signs[-1] != 0 and signs[-2] != 0 and signs[-3]== 0:
        signs[-1] = 0
        signs[1] = 0

    ######################### Identify runs of consecutive signals #####################

    copy = np.copy(signs)
    mask = copy != 0
    copy[mask] = 1
    change = np.diff(copy)

    # +1 signals to start, -1 signals to end, 100 is a singular point

    # Identify any singular points
    for x in range (0,num_steps-2):
        if change[x] == 1 and change[x+1] == -1:
            change[x+1] = 100
            change[x] = 0

    # Combine groups that are close together
    for x in range (0,num_steps-5):
        if change[x] == 100:
            for y in range (1,sensitivity+1):
                if change[x+y] == 1:
                    change[x] = 1
                    change[x+y] = 0
                    break
                elif change[x+y] == 100:
                    change[x] = 1
                    change[x+y] = -1
        if change[x] == -1:
            for y in range (1,sensitivity+1):
                if change[x+y] == 1:
                    change[x+y] = 0
                    change[x] = 0
                    break
                elif change[x+y] == 100:
                    change[x+y] = -1
                    change[x] = 0
                    break

    # Don't want the first/last entries be end/start signals
    if change[-1] == 1:
        change[-1] = 100
    if change[0] == -1:
        change[0] = 0

    indices = np.zeros(len(change))

    for x in range (0,num_steps-1):
        if change[x] == 1:
            indices[x] = x+1
        elif change[x] == -1 or change[x] == 100:
            indices[x] = x

    ######################### Get rid of runs of consecutive signals ###################

    signals = change[change != 0]
    indices = indices[indices != 0]
    signals = signals.astype(int)
    indices = indices.astype(int)

    # PROBLEM: Sometimes we start with an end signal and miss a start signal
    # This tries to deal with that:
    if signals[0] == -1:
        max = np.argmax(data[:indices[0]+1])
        min = np.argmin(data[:indices[0]+1])
        signs[0:indices[0]+1] = 0
        signs[min] = 1
        signs[max] = -1

    # PROBLEM: Sometimes there are two end signals in a row
    delete_elements = []
    for i in range (0, len(signals)-1):
        if signals[i] == -1 and signals[i+1] == -1:
            delete_elements.append(i)
    signals = np.delete(signals,delete_elements)
    indices = np.delete(indices,delete_elements)

    # If the edge signals are the same, pick the min/max in that range
    for i in range(0,len(signals)-1):
        if signals[i] == 1:
            start = indices[i]
            end = indices[i+1]
            if signs[start] == signs[end] or signs[start-1] == signs[end]:
                index = 0
                if signs[start-1] == 1 or signs[start-1] == -1:
                    start -= 1
                if signs[start] == 1:
                    index = np.argmin(data[start:end+1]) + start
                    signs[index] = 1
                else:
                    index = np.argmax(data[start:end+1]) + start
                    signs[index] = -1
                signs[start:index] = 0
                signs[index+1:end+1] = 0

    # If data[start:end] doesn't add any information, get rid of it
    for i in range (1, len(signals)-2):
        start = indices[i]
        end = indices[i+1]
        if signals[i] == 1:
            if (signs[start] == 1 or signs[start-1] == 1) and signs[end] == -1:
                if data[indices[i-1]] > data[start] and data[end] > data[indices[i+2]] and signs[indices[i-1]] != 1 and signs[indices[i+2]] != -1:
                    signs[start-1:end+1] = 0
            elif (signs[start] == -1 or signs[start-1] == -1) and signs[end] == 1:
                if data[indices[i-1]] < data[start] and data[end] < data[indices[i+2]] and signs[indices[i-1]] != -1 and signs[indices[i+2]] != 1:
                    signs[start-1:end+1] = 0

    # Deal with the side cases that can't be captured in the above
    if signals[0] == 1:
        if signs[indices[0]] == -1 and signs[indices[1]] == 1:
            if data[indices[2]] > data[indices[1]]:
                signs[indices[0]:indices[1]+1] = 0
                signs[indices[0]-1] = 0
            else:
                index = np.argmin(data[indices[0]:indices[1]+1]) + indices[0]
                signs[indices[0]:index] = 0
                signs[index+1:indices[1]+1] = 0
        if signs[indices[0]] == 1 and signs[indices[1]] == 1:
            index = np.argmin(data[indices[0]:indices[1]+1]) + indices[0]
            signs[indices[0]:index] = 0
            signs[index+1:indices[1]+1] = 0
        if signs[indices[0]] != -1 and signs[indices[1]] == -1:
            if data[indices[0]] - data[indices[1]] < 10:
                signs[indices[0]-1:indices[1]+1] = 0
            else:
                signs[indices[0]+1:indices[1]] = 0
    if signals[-1] == -1 and signs[indices[-2]] == 1 and signs[indices[-1]] == -1:
        if data[indices[-3]] > data[indices[-2]]:
            signs[indices[-2]:indices[-1]+1] = 0
    if signals[0] == -1 and signs[indices[0]] == -1:
        index = np.argmax(data[indices[0]:indices[1]+1]) + indices[0]
        signs[indices[0]:index] = 0
        signs[index+1:indices[1]+1] = 0

    return signs


def plot_signals(data, length, signs):
    """
    Here only for visual representation.
    """

    plt.clf()

    for x in range (0, length):
        if signs[x] == 1:
            plt.scatter(x, data[x],c='green')
        elif signs[x] == -1:
            plt.scatter(x, data[x],c='orange')

    plt.plot(data)
    plt.ylabel('stock price')
    plt.xlabel('day')
    plt.show()


############################## Putting it all together ############################

# This will be changed to be the output of the machine learning:
prediction = stockDataFinal[my_batch-3, :, (my_feature-1)*N_COMPANIES+my_company-1]
num_predictions = 150

# print("------------- Prediction -------------")
# print(prediction)
# print(len(prediction))
# print(type(prediction))

signs = determine_maxima_minima(prediction, num_predictions, sensitivity)
final_result = np.arange(2*num_predictions).reshape(2,num_predictions)
final_result[0,:] = prediction
final_result[1,:] = signs

# print("------------- Final Results -------------")
# print(final_result)
# print(len(final_result[0]))
# print(type(final_result))

day_list = []
order_list = []
price_list = []
for i in range(len(final_result[0])):
    if final_result[1, i]:
        day_list.append(i)
        price_list.append(prediction[i])
        if final_result[1, i] == 1:
            order_list.append("BUY")
        elif final_result[1, i] == -1:
            order_list.append("SELL")

orders = pd.DataFrame({'Day': day_list, 'Order': order_list, 'Price': price_list})

# plot_signals(prediction, num_predictions, signs)

