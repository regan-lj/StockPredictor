# IT3011_project: Stock Market Predictor

#### Our pipeline:

`stockData.py` -> The collection of data from yFinance<br/>
`data_preprocess.py` -> Getting rid of seasonality<br/>
`best_model'/'lin_model.py` -> The saved model / a script with the hyper-parameters configured to train and test this model (the trained model may differ on different machines due to pseudorandom component) <br/>
`buy_sell.py` -> The generation of buy/sell signals<br/>
`benchMark.py` -> Creation of the benchmark buy/sell signals<br/>
`evaluation.py` -> Comparison of our neural network with the original data and benchmark<br/>

#### Testing:

`trained_model_outputs/` -> Contains excerpts from the GPUs where models were trained with different hyper-parameters and their loss on the testing data was printed; validation loss and training loss were rarely close on our data, so the number of epochs were also trialled as a hyper-parameter <br/>
`model_results.txt` -> The predictions from the best model <br/>
`Evaluation Results` -> The practical evaluation using the benchmark as a comparison<br/>

#### Other files:

`obsolete/deep_learning_script*.py` -> The training script at the time of the presentation (configured for a previous dataset that is no longer used) <br/>
`deep_learning_RNN.py` -> The most recent training script, focused on tuning RNN hyper-parameters <br/>
`dynamic_buysell.py` -> Altered buysell that is able to generate signals on a rolling basis<br/>
`evaluationScript.py` -> Preliminary method of creating a benchmark<br/>
