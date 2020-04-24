# IT3011_project: Stock Market Predictor

#### Our pipeline:

`stockData.py` -> The collection of data from yFinance<br/>
`data_preprocess.py` -> Getting rid of seasonality<br/>
`lin_model.py` -> The neural network<br/>
`buy_sell.py` -> The generation of buy/slell signals<br/>
`evaluationScript.py` -> The benchmark<br/>
`evaluation.py` -> Comparison of our neural network with the benchmark<br/>

#### Testing:

`METHOD LOSS RESULTS.txt` -> Exploring how the hyperparameters affect the model<br/>
`model_results.txt` -> The best model predictions<br/>
`Evaluation Results` -> The practical evaluation using the benchmark as a comparison<br/>

#### Other files:

`chapter-15_demo_code*.py` -> Built upon the ideas in Chapter 15: Processing Sequences Using RNNs and CNNs<br/>
`deep_learning_script*.py` -> The neural network at the time of the presentation<br/>
`dynamic_buysell.py` -> Altered buysell that is able to generate signals on a rolling basis<br/>
`evaluationScript.py` -> A visual aid to see the benchmark<br/>
