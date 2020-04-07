import numpy as np
from buysell import *

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

num_predictions = 100
x1,x2 = generate_time_series(2,num_predictions)
array = x1[:num_predictions,0]

sensitivity = 4
present_index = 50
span = 5
stage = 1

signs = determine_maxima_minima(array, num_predictions, sensitivity)
generate_signals(array, signs, present_index, stage, span)
plot_signals(array, num_predictions, signs, present_index, stage, span)
