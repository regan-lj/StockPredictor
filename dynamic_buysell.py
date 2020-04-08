"""
This should take in the results from our machine learning.

The output is a 3 x time_period matrix that can be used for the GUI and for calculating net gain.
"""

# My imagination of how this would work:
# Potentially very wrong, not sure exactly how the ML works
#
# We have a certain number of days that we want to use our ML on. (the num of days in test data)
# As each day passes, we gain more information. As such, we now have a new prediction shifted one day forward.
# This prediction should in theory be fairly similar to the old prediction.
#
# So essentially we can loop through each day in our test data and determine a separate prediction for each day.
# For each iteration, I can take the past (say) 50 days of "known" data and a future 50 days of prediction and determine buy/sell/hold signals for each day.
# Then at the "present day", I can say whether we buy or sell with a given degree of uncertainty.
#
# We update the past known data and perform the ML on this updated data in every new iteration.

import numpy as np
from buysell import *

sensitivity = 4
span = 5

def generate_moving_signals(sensitivity, span, time_period):
    """
    Parameters
    ----------
    sensitivity : int
        How susceptible the buy/sell signals are to changes in price
        4 was ideal for mock data - may need to change with real data
    span : int
        How close present day has to be to a signal in order to activate it
        Too small, we run the risk of skipping important signals
        Too big, our signals will be less than optimal
    time_period : int
        The period of time over which we want to use this model
        NOTE: time_period is not necessarily the same as num_predictions

    Returns
    -------
    final_result
        An array with 3 rows and time_period columns:
        row1 = dates, row2 = prices, row3 = signals (+1: buy, -1: sell)
    """

    an_array_of_signals = np.zeros(time_period)
    stage = 1 # Start with a buy signal

    past_data = # The data that we know

    for day in range (0, time_period):

        result = # output of the machine learning

        modified_result = # modified output of machine learning : ideally a 2 x n_predictions matrix with dates in 1st row and prices in 2nd row

        present_index = # the last index of the past data

        data = # an array of past + future predictions (just the prices, no dates needed here)
        present_index = # the last index of the past data
        length = len(data)

        signs = determine_maxima_minima(data, length, sensitivity)
        event = generate_signals(data, signs, present_index, stage, span)

        if event == True:
            an_array_of_signals[day] = stage
            stage = -1*stage

        past_data = np.append(past_data, ) # need to update what we know

    final_result = # an_array_ of_signals appended as a 3rd row to result

    return final_result
