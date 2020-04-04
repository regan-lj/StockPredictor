import matplotlib.pyplot as plt
import numpy as np


def determine_buy_sell(prediction, n_steps, sensitivity):

    # Find local minima/maxima (very specific)
    buy = np.r_[True, prediction[1:] < prediction[:-1]] & np.r_[prediction[:-1] < prediction[1:], True]
    sell = np.r_[True, prediction[1:] > prediction[:-1]] & np.r_[prediction[:-1] > prediction[1:], True]

    signs = np.zeros(l) # Will be +1 for buy and -1 for sell

    # Map buy/sell to above vectors
    for x in range (0, l):
        if buy[x] == True:
            signs[x] = 1
        if sell[x] == True:
            signs[x] = -1

    # Ensures that the last signal is to sell
    if signs[-1] == 1:
        signs[-1] = 0

    # Ensures that the first signal is to buy
    if signs[0] == -1:
        signs[0] = 0


    ###################### Get rid of volatile and redundant values ####################


    # If there are 3 points close together with zeros on either side -> best of edges remains
    for x in range (2,l-2):
        if signs[x-2] == 0 and signs[x-1] != 0 and signs[x] != 0 and signs[x+1] != 0 and signs[x+2] == 0:
            if (signs[x] == 1 and prediction[x-1] > prediction[x+1]) or (signs[x] == -1 and prediction[x-1] < prediction[x+1]):
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
    for x in range (1,l-2):
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
    for x in range (0,l-2):
        if change[x] == 1 and change[x+1] == -1:
            change[x+1] = 100
            change[x] = 0

    # Combine groups that are close together
    for x in range (0,l-5):
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

    for x in range (0,l-1):
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
        max = np.argmax(prediction[:indices[0]+1])
        min = np.argmin(prediction[:indices[0]+1])
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
                    index = np.argmin(prediction[start:end+1]) + start
                    signs[index] = 1
                else:
                    index = np.argmax(prediction[start:end+1]) + start
                    signs[index] = -1
                signs[start:index] = 0
                signs[index+1:end+1] = 0

    # If prediction[start:end] doesn't add any information, get rid of it
    for i in range (1, len(signals)-2):
        start = indices[i]
        end = indices[i+1]
        if signals[i] == 1:
            if (signs[start] == 1 or signs[start-1] == 1) and signs[end] == -1:
                if prediction[indices[i-1]] > prediction[start] and prediction[end] > prediction[indices[i+2]] and signs[indices[i-1]] != 1 and signs[indices[i+2]] != -1:
                    signs[start-1:end+1] = 0
            elif (signs[start] == -1 or signs[start-1] == -1) and signs[end] == 1:
                if prediction[indices[i-1]] < prediction[start] and prediction[end] < prediction[indices[i+2]] and signs[indices[i-1]] != -1 and signs[indices[i+2]] != 1:
                    signs[start-1:end+1] = 0

    # Deal with the side cases that can't be captured in the above
    if signals[0] == 1 and signs[indices[0]] == -1 and signs[indices[1]] == 1:
        if prediction[indices[2]] > prediction[indices[1]]:
            signs[indices[0]:indices[1]+1] = 0
            signs[indices[0]-1] = 0
        else:
            index = np.argmin(prediction[indices[0]:indices[1]+1]) + indices[0]
            signs[indices[0]:index] = 0
            signs[index+1:indices[1]+1] = 0
    if signals[0] == 1 and signs[indices[0]] == 1 and signs[indices[1]] == 1:
        index = np.argmin(prediction[indices[0]:indices[1]+1]) + indices[0]
        signs[indices[0]:index] = 0
        signs[index+1:indices[1]+1] = 0
    if signals[0] == 1 and signs[indices[0]] != -1 and signs[indices[1]] == -1:
        if prediction[indices[0]] - prediction[indices[1]] < 10:
            signs[indices[0]-1:indices[1]+1] = 0
        else:
            signs[indices[0]+1:indices[1]] = 0
    if signals[-1] == -1 and signs[indices[-2]] == 1 and signs[indices[-1]] == -1:
        if prediction[indices[-3]] > prediction[indices[-2]]:
            signs[indices[-2]:indices[-1]+1] = 0
    if signals[0] == -1 and signs[indices[0]] == -1:
        index = np.argmax(prediction[indices[0]:indices[1]+1]) + indices[0]
        signs[indices[0]:index] = 0
        signs[index+1:indices[1]+1] = 0

    return prediction,signs


###################################### Plot data ###################################


def plot_signals(prediction, signs):

    for x in range (0, l):
        if signs[x] == 1:
            plt.scatter(x, prediction[x],c='green')
        if signs[x] == -1:
            plt.scatter(x, prediction[x],c='orange')

    plt.plot(x1)
    plt.ylabel('stock price')
    plt.show()


################################## Create mock data #################################


N_FEATURES = 1

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4,batch_size,1)
    time = np.linspace(0,1,n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # wave 2
    series += 0.1 * ( np.random.rand(batch_size,n_steps) - 0.5 ) # noise
    series = np.repeat(series.reshape(batch_size,n_steps,1), N_FEATURES, axis=2) # adds a third dimension for features
    return series.astype(np.float32)


####################################################################################


num_steps = 100
x1,x2 = generate_time_series(2,num_steps)
array = x1[:num_steps,0]
l = num_steps

prediction,signs = determine_buy_sell(array,l, 4)
plot_signals(array,signs)


# Next steps: adapt for a changing graph for when the days move forward and we gather more data

# Currently, I make sure that the first signal is always to buy - this needs to be changed for
# all day graphs after the first buy signal has been created. Needs to enforce alternation after
# this point.
