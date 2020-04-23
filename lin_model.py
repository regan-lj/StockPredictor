import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from stockData import stockDataFinal as rawData
from data_preprocess import series as networkData, reverse_differencing
import math
#######################################
import sys
sys.stdout = open('model_results', 'w')
#######################################
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10 # x, y
#######################################
#print("input data shape = ", networkData.shape)
#print("target data shape = ", rawData.shape)
stockDataFinal = networkData

print("DATA SHAPE : ", stockDataFinal.shape)

N_STEPS = stockDataFinal.shape[1]
INITIAL_BATCH_SIZE = stockDataFinal.shape[0]

N_PREDICTIONS = 99 # denoted by m in the pydocs
N_OUTPUT_FEATURES = 1
# data format: (dimension 1, dimension 2, dimension 3) = (BATCH_SIZE, N_TIME_STEPS, N_INPUT_FEATURES)

TRAINING_BATCH_SIZE = 13
VALIDATION_BATCH_SIZE = 3
TESTING_BATCH_SIZE = 4
N_INPUT_STEPS = N_STEPS - N_PREDICTIONS # denoted by n in the pydocs
stockDataFinal[:,:N_INPUT_STEPS,:] = networkData[:,:N_INPUT_STEPS,:]

# change parameters for the NN as well

for SEED in [420]: # 1 person try each seed
    for N_INPUT_NEURONS in [1]:
        for N_HIDDEN_NEURONS in range(N_INPUT_NEURONS, N_INPUT_NEURONS*2, math.ceil(N_INPUT_NEURONS/5)):
        #for N_HIDDEN_NEURONS in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
            for N_INPUT_FEATURES in [1]: # for each feature set (opening, high, low, closing, value)
                #if N_INPUT_FEATURES > 1:
                #    N_INPUT_FEATURES = N_INPUT_FEATURES - 1
                for N_EPOCHS in [10]:
                    tf.compat.v1.random.set_random_seed(SEED)

                    print()
                    print("SEED = ", SEED)
                    print("N_INPUT_NEURONS = ", N_INPUT_NEURONS)
                    print("N_HIDDEN_NEURONS = ", N_HIDDEN_NEURONS)
                    print("N_INPUT_FEATURES = ", N_INPUT_FEATURES)
                    print("N_EPOCHS = ", N_EPOCHS)
                    print()

                    assert INITIAL_BATCH_SIZE == TRAINING_BATCH_SIZE + VALIDATION_BATCH_SIZE + TESTING_BATCH_SIZE
                    assert N_OUTPUT_FEATURES <= N_INPUT_FEATURES

                    indices = (TRAINING_BATCH_SIZE,TRAINING_BATCH_SIZE+VALIDATION_BATCH_SIZE)

                    #series = generate_time_series(INITIAL_BATCH_SIZE, N_STEPS)
                    series = np.copy(stockDataFinal)
                    #get desired features
                    series = series[:,:,:N_INPUT_FEATURES]
                    x_train, y_train = series[:indices[0], :N_INPUT_STEPS], series[:indices[0], -N_PREDICTIONS:, :N_OUTPUT_FEATURES]
                    x_valid, y_valid = series[indices[0]:indices[1], :N_INPUT_STEPS], series[indices[0]:indices[1], -N_PREDICTIONS:, :N_OUTPUT_FEATURES]
                    x_test, y_test = series[indices[1]:, :N_INPUT_STEPS], series[indices[1]:, -N_PREDICTIONS:, :N_OUTPUT_FEATURES]

                    ####################################################################################################################################################################


                    ######################################################################### HELPER FUNCTIONS #########################################################################

                    def generate_time_series(batch_size, n_steps):
                        freq1, freq2, offsets1, offsets2 = np.random.rand(4,batch_size,1)
                        time = np.linspace(0,1,n_steps)
                        series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
                        series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # wave 2
                        series += 0.1 * ( np.random.rand(batch_size,n_steps) - 0.5 ) # noise
                        series = np.repeat(series.reshape(batch_size,n_steps,1), N_INPUT_FEATURES, axis=2) # adds a third dimension for features
                        return series.astype(np.float32)

                    def last_time_step_mse(y_true, y_pred):
                        """ Returns the MSE of predictions at the last time step; used when the following m time steps are predicted at every time step,
                            because only predictions made at the final time step will be useful for evaluation """
                        return keras.metrics.mean_squared_error(y_true[:,-1],y_pred[:,-1])

                    def get_sequence_to_vector_rnn_model(output_shape,input_shape=[None,N_INPUT_FEATURES]):
                        """ PARAMETERS
                                output_shape : the number of neurons in the output layer i.e. the size of the output vector
                                input_shape : [n_input_steps,n_input_features], default set to [None,N_INPUT_FEATURES] because an RNN can recur for any number of time steps from 0 to infinity
                            RETURNS
                                sequence-to-vector RNN model
                        """
                        layer1 = keras.layers.SimpleRNN(N_INPUT_NEURONS,return_sequences=True,input_shape=input_shape)
                        layer2 = keras.layers.SimpleRNN(N_HIDDEN_NEURONS)
                        layer3 = keras.layers.Dense(output_shape)
                        model = keras.models.Sequential([layer1,layer2,layer3])    
                        return model

                    def get_sequence_to_sequence_data(series):
                        """ Reorganises dataset for sequence_to_sequence predictions (i.e. predicting the next m time steps at each input time step)
                            PARAMETERS
                                series : the original dataset
                            RETURNS
                                The reorganised dataset"""
                        Y = np.empty((INITIAL_BATCH_SIZE,N_INPUT_STEPS,N_PREDICTIONS,N_OUTPUT_FEATURES))
                        # ^ At every one of our input time steps, we will predict the following N_PREDICTIONS time steps.
                        for step_ahead in range(1,N_PREDICTIONS+1):
                            Y[:,:,step_ahead-1,:N_OUTPUT_FEATURES] = series[:,step_ahead:step_ahead+N_INPUT_STEPS,:N_OUTPUT_FEATURES]
                        Y = Y.reshape(INITIAL_BATCH_SIZE,N_INPUT_STEPS,N_PREDICTIONS*N_OUTPUT_FEATURES)
                        return Y

                    def get_sequence_to_sequence_rnn_model(output_shape=N_PREDICTIONS*N_OUTPUT_FEATURES,layer_type=keras.layers.SimpleRNN,input_shape=[None,N_INPUT_FEATURES]):
                        """ PARAMETERS
                                output_shape : the number of neurons in the output layer i.e. the size of the output vector, default set to N_PREDICTIONS*N_OUTPUT_FEATURES
                                layer_type  : the layer type used to predict sequences, default set to a simple RNN layer
                                input_shape : [n_input_steps,n_input_features], default set to [None,N_INPUT_FEATURES] because an RNN can recur for any number of time steps from 0 to infinity
                            RETURNS
                                sequence-to-sequence RNN model
                        """
                        layer1 = layer_type(N_INPUT_NEURONS, return_sequences=True,input_shape=input_shape)
                        layer2 = layer_type(N_HIDDEN_NEURONS, return_sequences=True)
                        layer3 = keras.layers.Dense(output_shape)
                        #layer3 = keras.layers.TimeDistributed( keras.layers.Dense(output_shape) )
                        model = keras.models.Sequential([layer1,layer2,layer3])
                        return model

                    def train_sequence_to_sequence_rnn_model(model,x_train,y_train,x_valid,y_valid):
                        optimizer = keras.optimizers.Adam(lr=0.01)
                        model.compile(loss="mse",optimizer=optimizer, metrics=[last_time_step_mse])
                        training_history = model.fit(x_train,y_train,epochs=N_EPOCHS,validation_data=(x_valid,y_valid),verbose=0)
                        return training_history
                        
                    ######################################################################### BASELINE METRICS #########################################################################

                    def naive_forecasting(x_test, y_test):
                        """ Uses a naive prediction; assumes that all vectors from step n to step (n+m) are equivalent to the vector at time step (n-1).
                            PARAMETERS
                                x_test : test input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_test.shape = (TESTING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_test : test target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_test.shape = (TESTING_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                            RETURNS
                                 MSE of target data and naive prediction"""
                        y_pred = np.concatenate([x_test[:,-1].reshape((TESTING_BATCH_SIZE,1,N_INPUT_FEATURES)) for x in range(N_PREDICTIONS)],axis=1)
                        y_pred = y_pred[:,:,:N_OUTPUT_FEATURES]
                        return np.mean(keras.losses.mean_squared_error(y_test,y_pred)), y_pred

                    def linear_regression_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test):
                        """ Trains a linear regression model on the given input sequence.
                            PARAMETERS
                                x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_train : training target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_valid.shape = (TRAINING_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                                x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_valid : validation target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_valid.shape = (VALIDATION_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                                x_test : test input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_test.shape = (TESTING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_test : test target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_test.shape = (TESTING_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                            RETURNS
                                A history object containing the model's MSE after every epoch."""
                        y_train = y_train.reshape(TRAINING_BATCH_SIZE,N_PREDICTIONS*N_OUTPUT_FEATURES)
                        y_valid = y_valid.reshape(VALIDATION_BATCH_SIZE,N_PREDICTIONS*N_OUTPUT_FEATURES)
                        layer1 = keras.layers.Flatten(input_shape=[N_INPUT_STEPS,N_INPUT_FEATURES]) #input layer flattens each batch instance from [n_steps,n_input_features] to [n_steps*n_input_features]
                        layer2 = keras.layers.Dense(N_PREDICTIONS*N_OUTPUT_FEATURES) #fully connected layer solves combination of linear equations
                        model = keras.models.Sequential([layer1,layer2])
                        model.compile(loss="mse",optimizer="adam")
                        training_history = model.fit(x_train,y_train,epochs=N_EPOCHS,validation_data=(x_valid,y_valid),verbose=0)
                        y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
                        y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_PREDICTIONS,N_OUTPUT_FEATURES)
                        return training_history.history, y_pred, model

                    ####################################################### APPROACHES BETTER FOR DEALING WITH SHORTER SEQUENCES #######################################################

                    def rnn_iterative_forecasting(x_train,x_valid,x_test,series,indices):
                        """ Iterates from time step n to n+m, using sequence-to-vector forecasting to predict the target feature at step n+1 from the feature vector as step n;
                            trains a model to predict every input feature, predicts all features at a time step before moving to the next. Once all steps have been predicted,
                            the output features are extracted from all predicted time steps.
                            PARAMETERS
                                x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                series  : the complete initial dataset before partitioning, which we require to generate our targets for this method,
                                          series.shape = (INITIAL_BATCH_SIZE, N_STEPS, N_INPUT_FEATURES)
                                indices : a tuple containing the 2 indices required to partition our targets [into y_train and y_valid] once we have generated them.
                            RETURNS
                                A history object containing the model's MSE after every epoch."""

                        models = []
                        for feature in range(N_INPUT_FEATURES):
                            # model is only trained to predict the next time step (i.e. step n+1).
                            y_train, y_valid = series[:indices[0],-N_PREDICTIONS,feature], series[indices[0]:indices[1],-N_PREDICTIONS,feature]
                            m = get_sequence_to_vector_rnn_model(1)
                            m.compile(loss="mse",optimizer="adam")
                            m.fit(x_train,y_train,epochs=N_EPOCHS,validation_data=(x_valid,y_valid),verbose=0)
                            models.append(m)
                        full_sequence = np.zeros((TESTING_BATCH_SIZE,N_STEPS,N_INPUT_FEATURES))
                        full_sequence[:,:N_INPUT_STEPS,:] = x_test[:,:,:] # fill in the input time steps
                        for step in range(N_PREDICTIONS):
                            for feature in range(N_INPUT_FEATURES):
                                m = models[feature]
                                offset = N_INPUT_STEPS + step
                                #full_sequence[:,offset,N_OUTPUT_FEATURES:] = x_test[:,-1,N_OUTPUT_FEATURES:]
                                data_out = m.predict(full_sequence[:,step:offset,:],TESTING_BATCH_SIZE)
                                full_sequence[:,offset,feature] = data_out.reshape(TESTING_BATCH_SIZE) # output features span from index 0 to N_OUTPUT_STEPS
                        y_pred = full_sequence[:,-N_PREDICTIONS:,:N_OUTPUT_FEATURES]
                        return np.mean(keras.losses.mean_squared_error(y_test,y_pred)), y_pred

                    def rnn_vector_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test):
                        """ Uses sequence-to-vector forecasting to predict all time steps between n and n+m at time step n at once;
                            this version tries to predict every feature across all time steps.
                            PARAMETERS
                                x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_train : training target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_valid.shape = (TRAINING_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                                x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_valid : validation target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_valid.shape = (VALIDATION_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                                x_test : test input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_test.shape = (TESTING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                y_test : test target data, a batch of matrices which plot features over time (from time step n to step n+m)
                                          y_test.shape = (TESTING_BATCH_SIZE, N_PREDICTIONS, N_OUTPUT_FEATURES)
                            RETURNS
                                A history object containing the model's MSE after every epoch."""
                        y_train = y_train.reshape(TRAINING_BATCH_SIZE,N_PREDICTIONS*N_OUTPUT_FEATURES)
                        y_valid = y_valid.reshape(VALIDATION_BATCH_SIZE,N_PREDICTIONS*N_OUTPUT_FEATURES)
                        # ^ Each target is flattened into a batch of vectors to match the output layer (which will output a batch of vectors with size N_PREDICTIONS*N_OUTPUT_FEATURES)
                        model = get_sequence_to_vector_rnn_model(N_PREDICTIONS*N_OUTPUT_FEATURES)
                        model.compile(loss="mse",optimizer="adam")
                        training_history = model.fit(x_train,y_train,epochs=N_EPOCHS,validation_data=(x_valid,y_valid),verbose=0)
                        y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
                        y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_PREDICTIONS,N_OUTPUT_FEATURES)
                        return training_history.history, y_pred

                    # (only difference between the next 3 functions is layer_type in get_sequence_to_sequence_rnn_model function call)

                    def rnn_sequence_forecasting(x_train,x_valid,x_test,series,indices):
                        """ This version uses sequence-to-sequence forecasting to predict all remaining time steps [between n and n+m] at every time step between n and n+m.
                            By obtaining outputs at every time step, we have more error gradients flowing through the model which will stabilise and speed up training.
                            PARAMETERS
                                x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                series  : the complete initial dataset before partitioning, which we require to generate our targets for this method,
                                          series.shape = (INITIAL_BATCH_SIZE, N_STEPS, N_INPUT_FEATURES)
                                indices : a tuple containing the 2 indices required to partition our targets [into y_train and y_valid] once we have generated them.
                            RETURNS
                                A history object containing the model's MSE after every epoch."""

                        Y = get_sequence_to_sequence_data(series)
                        y_train, y_valid, y_test = Y[:indices[0]], Y[indices[0]:indices[1]], Y[indices[1]:]
                        #print(Y[0,0:3,:,:]) # For the first sample, for the first 3 time steps, displays predictions over the following m time steps.
                        model = get_sequence_to_sequence_rnn_model()
                        training_history = train_sequence_to_sequence_rnn_model(model,x_train,y_train,x_valid,y_valid)
                        y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
                        y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_INPUT_STEPS,N_PREDICTIONS,N_OUTPUT_FEATURES)[:,-1] #we only need the prediction from the last input time step
                        return training_history.history, y_pred

                    ####################################################### APPROACHES BETTER FOR DEALING WITH LONGER SEQUENCES #######################################################


                    def rnn_lstm_sequence_forecasting(x_train,x_valid,x_test,series,indices):
                        """ This version uses LSTM cells to perform sequence-to-sequence forecasting, so it should be better at learning/remembering long term patterns in the data.
                            PARAMETERS
                                x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                series  : the complete initial dataset before partitioning, which we require to generate our targets for this method,
                                          series.shape = (INITIAL_BATCH_SIZE, N_STEPS, N_INPUT_FEATURES)
                                indices : a tuple containing the 2 indices required to partition our targets [into y_train and y_valid] once we have generated them.
                            RETURNS
                                A history object containing the model's MSE after every epoch."""
                        Y = get_sequence_to_sequence_data(series)
                        y_train, y_valid, y_test = Y[:indices[0]], Y[indices[0]:indices[1]], Y[indices[1]:]
                        model = get_sequence_to_sequence_rnn_model(layer_type=keras.layers.LSTM)
                        training_history = train_sequence_to_sequence_rnn_model(model,x_train,y_train,x_valid,y_valid)
                        y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
                        y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_INPUT_STEPS,N_PREDICTIONS,N_OUTPUT_FEATURES)[:,-1] #we only need the prediction from the last input time step
                        return training_history.history, y_pred

                    def rnn_gru_sequence_forecasting(x_train,x_valid,x_test,series,indices):
                        """ This version uses GRU cells to perform sequence-to-sequence forecasting, so it should also be better at learning/remembering long term patterns in the data.
                            PARAMETERS
                                x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                          x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                                series  : the complete initial dataset before partitioning, which we require to generate our targets for this method,
                                          series.shape = (INITIAL_BATCH_SIZE, N_STEPS, N_INPUT_FEATURES)
                                indices : a tuple containing the 2 indices required to partition our targets [into y_train and y_valid] once we have generated them.
                            RETURNS
                                A history object containing the model's MSE after every epoch."""
                        Y = get_sequence_to_sequence_data(series)
                        y_train, y_valid, y_test = Y[:indices[0]], Y[indices[0]:indices[1]], Y[indices[1]:]
                        model = get_sequence_to_sequence_rnn_model(layer_type=keras.layers.GRU) 
                        training_history = train_sequence_to_sequence_rnn_model(model,x_train,y_train,x_valid,y_valid)
                        y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
                        y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_INPUT_STEPS,N_PREDICTIONS,N_OUTPUT_FEATURES)[:,-1] #we only need the prediction from the last input time step
                        return training_history.history, y_pred

                    ###################################################################################################################################################################
                    ###################################################################################################################################################################
                    ###################################################################################################################################################################
                    ###################################################################################################################################################################
                    ###################################################################################################################################################################

                    best = 99999999999999
                    best_m = None
                    best_n = ""

                    ###################################################################################################################################################################

                    naive_loss, pred = naive_forecasting(x_test, y_test) 
                    print("naive forecasting loss: ", naive_loss)

                    linear_model_history, pred, model = linear_regression_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test)
                    linear_loss = linear_model_history['val_loss'][-1] 
                    print("linear regression forecasting loss: ", linear_loss)

                    if linear_loss < best:
                        best = linear_loss
                        best_m = model
                        best_n = "linear_regression_forecasting"

print("\nBEST MODEL FOUND : ", best_n)
print("\nLOSS : ", best)

pred = pred.reshape((TESTING_BATCH_SIZE, N_PREDICTIONS))

print("\nBATCH 16 PREDICTION (time differenced): ", pred[0])
last_vals = rawData[16, 1, :N_OUTPUT_FEATURES]
print("\nBATCH 16 PREDICTION (reverse differenced): ", reverse_differencing(last_vals,pred[0]))

print("\nBATCH 17 PREDICTION (time differenced): ", pred[1])
last_vals = rawData[17, 1, :N_OUTPUT_FEATURES]
print("\nBATCH 17 PREDICTION (reverse differenced): ", reverse_differencing(last_vals,pred[1]))

print("\nBATCH 18 PREDICTION (time differenced): ", pred[2])
last_vals = rawData[18, 1, :N_OUTPUT_FEATURES]
print("\nBATCH 18 PREDICTION (reverse differenced): ", reverse_differencing(last_vals,pred[2]))

print("\nBATCH 19 PREDICTION (time differenced): ", pred[3])
last_vals = rawData[19, 1, :N_OUTPUT_FEATURES]
print("\nBATCH 19 PREDICTION (reverse differenced): ", reverse_differencing(last_vals,pred[3]), flush=True)
