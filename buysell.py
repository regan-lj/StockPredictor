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
import pandas as pd
from stockData import stockDataFinal,my_feature,my_company,N_COMPANIES
# from get_model_data import best_pred_reverse_differenced,actual_results

sensitivity = 4

my_batch = 16
best_pred_reverse_differenced = np.array([83.80945539, 87.44891262, 84.56918573, 84.45761088, 84.93989259, 85.17313449, 82.31829612, 81.32103364, 80.83008029, 81.12496503, 79.29618151, 81.24442573, 79.83642907, 80.00946383, 81.89280216, 80.91176616, 80.1266024,  82.04627, 82.40667059, 82.44633352, 80.62800002, 81.50900465, 82.16330599, 85.65083193, 86.77260565, 85.65220809, 84.61126828, 80.94063449, 79.27899038, 75.73119557, 74.85036659, 75.64804482, 74.25983941, 74.33903309, 74.06043169, 74.89475229, 73.09936705, 74.20081237, 75.34962156, 76.45481267, 76.27245117, 76.10905319, 75.40805155, 74.20071262, 73.60326802, 73.59344163, 72.72227844, 74.26269814, 74.26812464, 77.13801629, 76.49983097, 74.82519258, 74.75304732, 77.41731892, 77.84767059, 79.49607364, 81.35767785, 79.46644572, 80.92331532, 81.9861081, 80.25521318, 80.41677785, 83.60088015, 82.4470743, 82.00728465, 82.69828845, 82.73671921, 83.05304983, 83.19737863, 82.90608095, 80.96619784, 83.22442781, 81.09872877, 80.69655629, 80.63707945, 81.91984293, 79.18474338, 79.19871382, 77.4377294,  74.5272551,  74.22089215, 75.37900229, 77.55781121, 78.00630571, 77.64852579, 79.32843991, 80.90539665, 80.72333646, 83.89243412, 85.18334138, 83.0516597, 80.07159078, 81.28583944, 82.06434578, 82.6918984,  84.823165, 83.31829256, 82.48813021, 84.03249752, 83.73434755])

# my_batch = 17
# best_pred_reverse_differenced = np.array([95.22874618, 93.95749235, 91.52717996, 94.41334772, 94.33929911, 95.67412594, 97.23650983, 95.13527516, 96.86047688, 98.84263816, 98.04681137, 95.28308681, 94.56405565, 92.59205148, 92.89729697, 92.69320855, 95.65315971, 95.29505351, 95.30549167, 92.74390458, 92.73320371, 92.19584465, 91.61337059, 92.95919746, 93.51191049, 94.28639739, 94.9967702,  95.34797907, 94.91615763, 91.84868589, 93.83368232, 93.96429315, 94.0350786,  95.18302436, 95.19150624, 96.75654755, 96.77789854, 96.77820065, 99.41040947, 97.45285406, 95.64531126, 95.76549288, 93.18596145, 92.26114174, 95.09825703, 95.66356339, 95.34833085, 96.80804323, 95.79008984, 94.73932134, 93.66981398, 93.16718989, 91.97031277, 90.37008172, 91.0973069,  90.26774275, 90.76938783, 89.37214505, 88.82603347, 90.03999936, 90.65938985, 90.72281468, 89.43935955, 92.03389585, 93.80039179, 92.69454098, 92.35621288, 93.61970308, 93.68476546, 92.17686212, 93.89070094, 93.55190817, 95.88434592, 94.81040838, 92.93064228, 91.5498127,  93.21236959, 94.31929603, 96.54265204, 95.5149999,  95.13713431, 93.91908467, 93.16195124, 93.15925398, 88.29325905, 88.68216335, 88.71599677, 86.82811526, 87.73307893, 84.78343603, 82.78143999, 80.25415251, 81.14996252, 80.43426904, 79.58391711, 79.69300427, 81.03697397, 79.47155644, 80.04939725, 77.76615979])

# my_batch = 18
# best_pred_reverse_differenced = np.array([151.45074725, 149.18149328, 151.71832561, 150.16431546, 152.18840384, 151.81937629, 147.82379395, 152.18640524, 160.05717665, 158.71116751, 167.49784297, 166.01948744, 159.13826519, 150.41296822, 145.74895197, 147.33629435, 136.70662612, 140.00735539, 136.36768049, 140.82483333, 141.50265747, 143.28458112, 134.20421499, 130.58104175, 122.27387375, 120.94884866, 109.58363432, 113.17496151, 112.57906091, 103.16909349, 100.48814023, 95.54676354, 90.19950926, 91.21478987, 88.707582,  85.45631742, 82.88782287, 80.10905886, 69.14671564, 73.36999035,  75.23729002, 77.83947003, 79.31402469, 72.11072659, 79.50899863,  82.96873569, 89.60800076, 87.38080907, 95.79349256, 103.21980405, 101.93776429, 104.40554464, 99.74676216, 99.89868045, 103.08985686, 109.79103684, 116.5101068, 126.87206531, 126.77938627, 126.54405908, 125.60324917, 130.41356764, 125.99700413, 127.93670843, 122.48647783, 122.02852861, 122.0480743, 124.06097465, 126.74294143, 125.76580899, 132.2791013, 126.06052392, 134.44671338, 139.81162875, 139.8926243, 134.03833238, 130.74159757, 123.47172634, 122.74742488, 128.90023211, 126.02376989, 135.68908361, 133.24241761, 135.70729283, 136.69657842, 135.71272025, 128.21385894, 124.33370481, 119.03057466, 120.08093903, 119.38780872, 122.65824787, 123.77432445, 117.1841021, 115.00028995, 105.20082382, 106.08593891, 101.77041719, 110.93281171, 110.31690071])

# my_batch = 19
# best_pred_reverse_differenced = np.array([215.69648337, 211.8429637,  213.89439273, 218.46867728, 213.00536418, 223.65947223, 220.59942079, 218.00834274, 217.42216098, 217.25351308, 217.98749749, 216.82477085, 218.66693155, 219.14622499, 219.65252842, 214.36844029, 213.16758932, 204.67931665, 207.55553354, 206.37414099, 209.35868885, 211.52940871, 219.96159483, 215.07264496, 219.92837407, 226.62228371, 237.37536313, 238.44013669, 242.11118723, 243.12700047, 246.37671508, 245.30968048, 237.88709833, 238.69294192, 235.24691369, 231.65374185, 231.48580582, 225.18673261, 217.1952699, 220.4701875, 224.80210479, 217.71561654, 206.81121762, 198.71518739, 192.30388625, 185.40273078, 195.19224437, 196.22163387, 196.46856758, 194.49488625, 200.55665049, 204.2217277, 205.7435585, 213.53968009, 215.92799291, 213.51524028, 219.73607406, 217.89477023, 216.02822813, 222.41657481, 223.18279603, 221.81360498, 212.086826, 214.86057439, 205.88977399, 204.36454597, 203.47258803, 199.25783536, 200.79372105, 201.200564, 203.29127797, 213.6986514, 214.2023963, 209.17045775, 212.8616499, 208.86545554, 225.02130118, 221.6963903, 215.90561882, 231.33001438, 234.22221318, 235.5409036, 238.26041415, 242.43553975, 246.30072621, 246.24801096, 248.0299786, 236.44175896, 239.04127011, 241.88424429, 240.30216607, 244.66628036, 237.1579729, 230.38164148, 234.58815917, 228.30025539, 228.18159652, 233.89606977, 235.33839059, 234.99434263])



def fix_problems(data, num_steps, signs):
    """
    Fixes any issues with determine_maxima_minima.
    Implemented separately simply for ease.
    """

    new_signs = np.zeros(num_steps)
    num_indices = 0

    # Take all the sell signals
    for k in range(0,num_steps):
        if signs[k] == -1:
            new_signs[k] = -1
            num_indices += 1

    # Find the minmas between the sell signals
    sell_indices = np.zeros(num_indices)
    step = 0
    for k in range(0,num_steps):
        if new_signs[k] == -1:
            sell_indices[step] = k
            step += 1
    sell_indices = sell_indices.astype(int)
    print("sell_indices",sell_indices)
    for k in range(0,num_indices-1):
        min = sell_indices[k] + np.argmin(data[sell_indices[k]:sell_indices[k+1]-1])
        new_signs[min] = 1
        num_indices += 1

    new_signs[np.argmin(data[0:sell_indices[0]])] = 1

    print(new_signs)

    steps = np.arange(num_steps)
    print(steps)

    maximas = steps[new_signs == -1]
    print(maximas)

    minimas = steps[new_signs == 1]
    print(minimas)


    return new_signs


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

    # PROBLEM: Sometimes there are two buy signals or sell signals in a row
    # PROBLEM: A few signals close together
    signs = fix_problems(data, num_steps, signs)

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
# prediction = stockDataFinal[my_batch-6, :, (my_feature-1)*N_COMPANIES+my_company-1]
# num_predictions = 150

prediction = best_pred_reverse_differenced
num_predictions = len(prediction)

signs = determine_maxima_minima(prediction, num_predictions, sensitivity)
final_result = np.arange(2*num_predictions).reshape(2,num_predictions)
actual_results_MAIN = stockDataFinal[my_batch-1, :, (my_feature-1)*N_COMPANIES+my_company-1][-100:]

print("------------- Actual Results -------------")
print(actual_results_MAIN)

final_result[0,:] = actual_results_MAIN
final_result[1,:] = signs



# ML Prediction (save into final_results[0])

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
        price_list.append(actual_results_MAIN[i])
        if final_result[1, i] == 1:
            order_list.append("BUY")
        elif final_result[1, i] == -1:
            order_list.append("SELL")

orders = pd.DataFrame({'Day': day_list, 'Order': order_list, 'Price': price_list})

# plot_signals(prediction, num_predictions, signs)
