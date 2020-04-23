import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from stockData import stockDataFinal as targetData
from data_preprocess import series as inputData
#######################################
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10 # x, y
#######################################
print("input data shape = ", inputData.shape)
print("target data shape = ", targetData.shape)
inputData = inputData[:-1]###
targetData = targetData[:-1]###
targetData = inputData ###
stockDataFinal = targetData

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

indices = (TRAINING_BATCH_SIZE,TRAINING_BATCH_SIZE+VALIDATION_BATCH_SIZE)

#SEED = 420
#EPOCH = 100
N_INPUT_FEATURES = 41

#series = generate_time_series(INITIAL_BATCH_SIZE, N_STEPS)
series = np.copy(stockDataFinal)
#get desired features
series = series[:,:,:N_INPUT_FEATURES]
x_train, y_train = series[:indices[0], :N_INPUT_STEPS], series[:indices[0], -N_PREDICTIONS:, :N_OUTPUT_FEATURES]
x_valid, y_valid = series[indices[0]:indices[1], :N_INPUT_STEPS], series[indices[0]:indices[1], -N_PREDICTIONS:, :N_OUTPUT_FEATURES]
x_test, y_test = series[indices[1]:, :N_INPUT_STEPS], series[indices[1]:, -N_PREDICTIONS:, :N_OUTPUT_FEATURES]

class Plotter:

    """Responsible for plotting data"""

    def __init__(self, n_predictions):
        self.fig = plt.axes()
        self.legend = []
        self.n_predictions = n_predictions
        self.fig.set_xticks(range(1,n_predictions))
        self.TARGET_FEATURE = 0 # The target feature that will be plotted on the graph

    def plot(self,y_data,name,batch_size,batch_id=0):
        """ Plots the target feature from a batch instance of the predicted data to the current figure.
            PARAMETERS
                y_data  : a batch of matrices which plot features over time (from time step 0 to step n-1),
                          y_data.shape = (BATCH_SIZE, N_PREDICTIONS)
                name    : string, a label for our y_data on the graph
                batch_id : integer from 0 to (BATCH_SIZE-1), denotes which batch instance to plot from the y_data"""
        y_data = y_data[:,:,self.TARGET_FEATURE]
        plot_data = y_data.reshape((batch_size,self.n_predictions))[batch_id]
        self.fig.plot(plot_data)
        self.legend.append(name)

    def show(self):
        self.fig.legend(self.legend)
        plt.show()

def last_time_step_mse(y_true, y_pred):
    """ Returns the MSE of predictions at the last time step; used when the following m time steps are predicted at every time step,
        because only predictions made at the final time step will be useful for evaluation """
    return keras.metrics.mean_squared_error(y_true[:,-1],y_pred[:,-1])

model = tf.keras.models.load_model("best_model",compile=False)#custom_objects={"last_time_step_mse": last_time_step_mse})
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse",optimizer=optimizer, metrics=[last_time_step_mse])
y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
print("y_pred.shape = ", y_pred.shape)
y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_PREDICTIONS,N_OUTPUT_FEATURES)

y_loss = np.mean(keras.losses.mean_squared_error(y_test,y_pred))
print("model forecasting loss: ", y_loss)

pltr = Plotter(N_PREDICTIONS)
pltr.plot(y_test,"target data",TESTING_BATCH_SIZE)

pltr.plot(y_pred,"output data",TESTING_BATCH_SIZE)

pltr.fig.legend(pltr.legend)
pltr.fig.figure.savefig("figures/MODEL_RESULTS")
pltr.show()
plt.close()
