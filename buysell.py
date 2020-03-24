import matplotlib.pyplot as plt
import numpy as np
#from scipy.signal import argrelextrema

N_FEATURES = 1

##########################################################

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4,batch_size,1)
    time = np.linspace(0,1,n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # wave 2
    series += 0.1 * ( np.random.rand(batch_size,n_steps) - 0.5 ) # noise
    series = np.repeat(series.reshape(batch_size,n_steps,1), N_FEATURES, axis=2) # adds a third dimension for features
    return series.astype(np.float32)

##########################################################

# Create mock data
num_steps = 100
x1,x2 = generate_time_series(2,num_steps)
array = x1[:num_steps,0]
l = num_steps

#array = np.convolve(array)

# Find local minma/maxima (very specific)
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

# Get rid of volatile values
for x in range (0, l-2):
    # Two points next to each other -> get rid of both
    # There doesn't exist a case where this will be true while also including a local maxima/minima
    if dec[x-1] == 0 and dec[x] != 0 and dec[x+1] != 0 and dec[x+2] == 0:
        dec[x] = 0
        dec[x+1] = 0
        decval[x] = 0
        decval[x+1] = 0
    # Three points by close together and by themselves -> keep one
    # Middle point is a local minima:
    if dec[x-1] == 1 and dec[x] == -1 and dec[x+1] == 1:
        dec[x] = 0
        decval[x] = 0
        if decval[x-1] < decval[x+1]:
            dec[x+1] = 0
            decval[x+1] = 0
        else:
            dec[x-1] = 0
            decval[x-1] = 0
    # Middle point is a local maxima:
    if dec[x-1] == -1 and dec[x] == 1 and dec[x+1] == -1:
        dec[x] = 0
        decval[x] = 0
        if decval[x-1] > decval[x+1]:
            dec[x+1] = 0
            decval[x+1] = 0
        else:
            dec[x-1] = 0
            decval[x-1] = 0

# Get rid of redundant points
# Still to complete
diff = np.diff(decval)

# Ensure that the first signal is buy and that the final signal is sell
# Still to complete

# Plot
for x in range (0, l):
    if dec[x] == 1:
        plt.scatter(x,array[x], c='green')
    if dec[x] == -1:
        plt.scatter(x,array[x], c='orange')

plt.plot(x1)
plt.ylabel('stock price')
plt.show()
