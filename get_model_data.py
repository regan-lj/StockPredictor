import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from stockData import stockDataFinal as targetData
from data_preprocess import series as inputData, reverse_differencing
#######################################
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10 # x, y
#######################################
print("input data shape = ", inputData.shape)
print("target data shape = ", targetData.shape)
inputData = inputData[:-1]###
targetData = targetData[:-1]###
stockDataFinal = inputData

print(stockDataFinal.shape)

N_STEPS = stockDataFinal.shape[1]
INITIAL_BATCH_SIZE = stockDataFinal.shape[0]

N_PREDICTIONS = 49 # denoted by m in the pydocs
N_OUTPUT_FEATURES = 1
# data format: (dimension 1, dimension 2, dimension 3) = (BATCH_SIZE, N_TIME_STEPS, N_INPUT_FEATURES)

TRAINING_BATCH_SIZE = 5
VALIDATION_BATCH_SIZE = 2
TESTING_BATCH_SIZE = 1
N_INPUT_STEPS = N_STEPS - N_PREDICTIONS # denoted by n in the pydocs
stockDataFinal[:,:N_INPUT_STEPS,:] = inputData[:,:N_INPUT_STEPS,:]

best_pred = np.array([-0.5893711, -0.4509366,  -0.53443784, -0.12847176, -0.3840047,  -0.281577,
 -0.43421674, -0.17125215, -0.42712444, -0.18797985, -0.4827967,  -0.33266687,
  0.11259076,  0.23752496,  0.17461157,  0.5008496,   0.45543528,  0.47983435,
  0.62416154,  0.23784126, -0.12254406,  0.35976186, 0.40632877, -0.0011216,
  0.19307123,  0.04603142, -0.38634998, -0.16371265, -0.23133239,  0.1254877,
 -0.05665697, -0.44663203, -0.32472008, -0.7864444,  -0.22138996, -0.6968259,
 -0.28179577, -0.07801341, -0.69729024, -0.73534954, -0.472994,   -0.6099597,
 -0.14156011, -0.29058978, -0.5914483,  -0.38043922, -0.5009831,   0.11425107,
 -0.5877607])

best_pred_reverse_differenced = np.array([85.24063075, 84.65125966, 84.20032308, 83.66588524, 83.53741348, 83.15340877, 82.87183177, 82.43761504, 82.26636289, 81.83923845,
                                 81.6512586,  81.1684619, 80.83579503, 80.94838579, 81.18591075, 81.36052231, 81.86137192, 82.3168072, 82.79664154, 83.42080308,
                                 83.65864435, 83.53610029, 83.89586215, 84.30219092, 84.30106932, 84.49414056, 84.54017197, 84.153822, 83.99010934, 83.75877695,
                                 83.88426465, 83.82760768, 83.38097565, 83.05625557, 82.26981114, 82.04842118, 81.35159526, 81.06979949, 80.99178608, 80.29449584,
                                 79.55914631, 79.08615231, 78.47619258, 78.33463248, 78.0440427, 77.45259439, 77.07215517, 76.57117205, 76.68542312, 76.09766243])
      
actual_results = targetData[-1,-(N_PREDICTIONS+1):,N_OUTPUT_FEATURES]
