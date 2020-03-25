import matplotlib.pyplot as plt
import numpy as np


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

num_steps = 100
x1,x2 = generate_time_series(2,num_steps)
array = x1[:num_steps,0]
l = num_steps


#####################################################################################


# Find local minima/maxima (very specific)
buy = np.r_[True, array[1:] < array[:-1]] & np.r_[array[:-1] < array[1:], True]
sell = np.r_[True, array[1:] > array[:-1]] & np.r_[array[:-1] > array[1:], True]

dec = np.zeros(l) # Will be +1 for buy and -1 for sell
decval = np.zeros(l) # Will have the stock price (only for days when we buy/sell)

# Map buy/sell to above vectors
for x in range (0, l):
    if buy[x] == True:
        dec[x] = 1
        decval[x] = array[x]
    if sell[x] == True:
        dec[x] = -1
        decval[x] = array[x]

# Ensures that the last signal is to sell
if dec[-1] == 1:
    dec[-1] = 0

# Ensures that the first signal is to buy
if dec[0] == -1:
    dec[0] = 0


###################### Get rid of volatile and redundant values ####################


# If there are 3 points close together with zeros on either side -> best of edges remains
for x in range (2,l-2):
    if dec[x-2] == 0 and dec[x-1] != 0 and dec[x] != 0 and dec[x+1] != 0 and dec[x+2] == 0:
        if (dec[x] == 1 and decval[x-1] > decval[x+1]) or (dec[x] == -1 and decval[x-1] < decval[x+1]):
            dec[x+1] = 0
        else:
            dec[x-1] = 0
        dec[x] = 0


# If there are 2 points close together with zeros on either side -> make both zero
# Impossible to be a local maxima/minima
for x in range (1,l-2):
    if dec[x-1] == 0 and dec[x] != 0 and dec[x+1] != 0 and dec[x+2] == 0:
        dec[x] = 0
        dec[x+1] = 0

if dec[0] != 0 and dec[1] != 0 and dec[2] == 0:
    dec[0] = 0
    dec[1] = 0

if dec[-1] != 0 and dec[-2] != 0 and dec[-3]== 0:
    dec[-1] = 0
    dec[1] = 0

#np.split(array, np.where(np.diff(copy))[0]+1)

# copy = np.copy(dec)
# mask = copy != 0
# copy[mask] = 1
# change = np.diff(copy)
# diff = 0.01
#
# print(dec)
# print(change)
#
# # -1 signals end, +1 signals start
# start = 0
# end = 100
# startindex = -1
# endindex = -1
# for x in range (0,l-1):
#     if change[x] == 1:
#         start = decval[x+1]
#         startindex = x + 1
#     if change[x] == -1:
#         end = decval[x+1]
#         endindex = x + 1
#         if abs(end - start) < diff:
#             dec[startindex:endindex] = 0
#         start = 0
#         end = 100
#
# print(dec)


###################################### Plot data ###################################

# Plot the buy/sell signals
for x in range (0, l):
    if dec[x] == 1:
        plt.scatter(x,array[x], c='green')
    if dec[x] == -1:
        plt.scatter(x,array[x], c='orange')

plt.plot(x1)
plt.ylabel('stock price')
plt.show()
