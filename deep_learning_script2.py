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
targetData = inputData ###
stockDataFinal = targetData

print(stockDataFinal.shape)

N_STEPS = stockDataFinal.shape[1]
INITIAL_BATCH_SIZE = stockDataFinal.shape[0]

N_PREDICTIONS = 49 # denoted by m in the pydocs
N_OUTPUT_FEATURES = 1
# data format: (dimension 1, dimension 2, dimension 3) = (BATCH_SIZE, N_TIME_STEPS, N_INPUT_FEATURES)

TRAINING_BATCH_SIZE = 6
VALIDATION_BATCH_SIZE = 2
TESTING_BATCH_SIZE = 1
N_INPUT_STEPS = N_STEPS - N_PREDICTIONS # denoted by n in the
stockDataFinal[:,:N_INPUT_STEPS,:] = inputData[:,:N_INPUT_STEPS,:]
            

for SEED in [1234, 777, 69, 420]: # 1 person try each seed
    for N_INPUT_FEATURES in range(41, 46, 5):
        for N_EPOCHS in [100, 200, 300, 400, 500, 600, 800, 1000]:
            tf.compat.v1.random.set_random_seed(SEED)

            print()
            print("SEED = ", SEED)
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
                layer1 = keras.layers.SimpleRNN(20,return_sequences=True,input_shape=input_shape)
                layer2 = keras.layers.SimpleRNN(20)
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
                layer1 = layer_type(20, return_sequences=True,input_shape=input_shape)
                layer2 = layer_type(20, return_sequences=True)
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
                return training_history.history, y_pred

            ####################################################### APPROACHES BETTER FOR DEALING WITH SHORTER SEQUENCES #######################################################

            def rnn_iterative_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test):
                """ Iterates from time step n to n+m, using sequence-to-vector forecasting to predict the target feature at step n+1 from the feature vector as step n;
                    this version assumes that every input feature which is not predicted remains constant to its value at the last input time step.
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
                        MSE of validation target data and the accumulation of predictions"""

                y_train = y_train[:,0] # model is only trained to predict the next time step (i.e. step n+1).
                model = get_sequence_to_vector_rnn_model(N_OUTPUT_FEATURES)
                model.compile(loss="mse",optimizer="adam")
                model.fit(x_train,y_train,epochs=N_EPOCHS,validation_data=(x_valid,y_valid),verbose=0)
                full_sequence = np.zeros((TESTING_BATCH_SIZE,N_STEPS,N_INPUT_FEATURES))
                full_sequence[:,:N_INPUT_STEPS,:] = x_testing[:,:,:] # fill in the input time steps
                for step in range(N_PREDICTIONS):
                    offset = N_INPUT_STEPS + step
                    full_sequence[:,offset,N_OUTPUT_FEATURES:] = x_testing[:,-1,N_OUTPUT_FEATURES:] # insert Naive input features (constant from the last input step)
                    data_out = model.predict(full_sequence[:,step:offset,:],TESTING_BATCH_SIZE)
                    full_sequence[:,offset,:N_OUTPUT_FEATURES] = data_out.reshape(TESTING_BATCH_SIZE,N_OUTPUT_FEATURES) # output features span from index 0 to N_OUTPUT_STEPS
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

            def cnn_vector_forecasting(x_train,x_valid,x_test,series,indices):
                """ Uses a simplified WaveNet CNN architecture
                    PARAMETERS
                        x_train : training input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                  x_valid.shape = (TRAINING_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                        x_valid : validation input data, a batch of matrices which plot features over time (from time step 0 to step n-1),
                                  x_valid.shape = (VALIDATION_BATCH_SIZE, N_INPUT_STEPS, N_INPUT_FEATURES)
                        series  : the complete initial dataset before partitioning, which we require to generate our targets for this method,
                                  series.shape = (INITIAL_BATCH_SIZE, N_STEPS, N_INPUT_FEATURES)
                        indices : a tuple containing the 2 indices required to partition our targets [into y_train and y_valid] once we have generated them,
                    RETURNS
                        A history object containing the model's MSE after every epoch."""
                Y = get_sequence_to_sequence_data(series)
                y_train, y_valid, y_test = Y[:indices[0]], Y[indices[0]:indices[1]], Y[indices[1]:]
                ### CAN CHANGE THIS : ###
                hidden_layer_dilations = (1,2,4,8,1,2,4,8)
                #########################
                layers = [keras.layers.InputLayer(input_shape=[None,N_INPUT_FEATURES])]
                ### CAN CHANGE THIS : ###
                layers.extend([keras.layers.Conv1D(filters=N_PREDICTIONS,kernel_size=2,padding="causal",activation="relu",dilation_rate=rate) for rate in hidden_layer_dilations])
                #########################
                layers.extend([keras.layers.Conv1D(filters=N_PREDICTIONS*N_OUTPUT_FEATURES,kernel_size=1)])
                model = keras.models.Sequential(layers)
                training_history = train_sequence_to_sequence_rnn_model(model,x_train,y_train,x_valid,y_valid)
                y_pred = model.predict(x_test, TESTING_BATCH_SIZE)
                y_pred = y_pred.reshape(TESTING_BATCH_SIZE,N_INPUT_STEPS,N_PREDICTIONS,N_OUTPUT_FEATURES)[:,-1] #we only need the prediction from the last input time step
                return training_history.history, y_pred

            ###################################################################################################################################################################
            ###################################################################################################################################################################
            ###################################################################################################################################################################
            ###################################################################################################################################################################
            ###################################################################################################################################################################

            pltr = Plotter(N_PREDICTIONS)
            pltr.plot(y_test,"target data",TESTING_BATCH_SIZE)

            ###################################################################################################################################################################

            naive_loss, pred = naive_forecasting(x_test, y_test) 
            print("naive forecasting loss: ", naive_loss)
            pltr.plot(pred,"naive prediction",TESTING_BATCH_SIZE)

            linear_model_history, pred = linear_regression_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test)
            linear_loss = linear_model_history['val_loss'][-1] 
            print("linear regression forecasting loss: ", linear_loss)
            #pltr.plot(pred,"linear prediction",TESTING_BATCH_SIZE)
            """
            rnn_iterative_loss, pred = rnn_iterative_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test) 
            print("deep rnn iterative forecasting loss: ", rnn_iterative_loss)
            pltr.plot(pred,"iterative prediction",TESTING_BATCH_SIZE)
            """
            rnn_vector_model_history, pred = rnn_vector_forecasting(x_train,y_train,x_valid,y_valid,x_test,y_test)
            rnn_vector_loss = np.mean(keras.losses.mean_squared_error(y_test,pred)) #rnn_vector_model_history['val_loss'][-1] 
            print("deep rnn vector forecasting loss: ", rnn_vector_loss)
            pltr.plot(pred,"vector prediction",TESTING_BATCH_SIZE)

            rnn_sequence_model_history, pred = rnn_sequence_forecasting(x_train,x_valid,x_test,series,indices)
            rnn_sequence_loss = np.mean(keras.losses.mean_squared_error(y_test,pred)) #rnn_sequence_model_history['val_loss'][-1]
            print("deep rnn sequence forecasting loss: ", rnn_sequence_loss)
            pltr.plot(pred,"sequence prediction",TESTING_BATCH_SIZE)

            ###################################################################################################################################################################

            rnn_lstm_model_history, pred = rnn_lstm_sequence_forecasting(x_train,x_valid,x_test,series,indices)
            rnn_lstm_loss = np.mean(keras.losses.mean_squared_error(y_test,pred)) #rnn_lstm_model_history['val_loss'][-1]
            print("deep rnn lstm sequence forecasting loss: ", rnn_lstm_loss)
            pltr.plot(pred,"ltsm sequence prediction",TESTING_BATCH_SIZE)

            rnn_gru_model_history, pred = rnn_gru_sequence_forecasting(x_train,x_valid,x_test,series,indices)
            rnn_gru_loss = np.mean(keras.losses.mean_squared_error(y_test,pred)) #rnn_gru_model_history['val_loss'][-1] 
            print("deep rnn gru sequence forecasting loss: ", rnn_gru_loss)
            pltr.plot(pred,"gru sequence prediction",TESTING_BATCH_SIZE)

            cnn_vector_model_history, pred = cnn_vector_forecasting(x_train,x_valid,x_test,series,indices)
            cnn_vector_loss = np.mean(keras.losses.mean_squared_error(y_test,pred)) #cnn_vector_model_history['val_loss'][-1] 
            print("cnn vector forecasting loss: ", cnn_vector_loss)
            pltr.plot(pred,"cnn sequence prediction",TESTING_BATCH_SIZE)

            name = "SEED="+str(SEED)+"-N_INPUT_FEATURES="+str(N_INPUT_FEATURES)+"-N_EPOCHS="+str(N_EPOCHS)
            pltr.fig.legend(pltr.legend)
            pltr.fig.figure.savefig("figures/"+name)
            plt.close()
            #pltr.show()

