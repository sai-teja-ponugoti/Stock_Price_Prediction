# Stock Price Prediction
Using time series data to predict the furture stock price using previous data


### Overview 
* Time Series is a collection of indexed data points based on the time during which they were collected. The data is most often recorded at regular time intervals.
* In practise, predicting future values for the time series is a very common problem. Predicting next week's weather, stock prices, tomorrow's Bitcoins price, the amount of your Chrismas sales and potential heart disease are common examples of this.

* Recurrent neural networks ( RNNs) may predict, or classify, the next value(s) in a series. A series is stored as a matrix, where each row is a descriptive vector of a function. The order of the rows in the matrix is of course essential.

* Time Series is just one type of a sequence. We’ll have to cut the Time Series into smaller sequences, so our RNN models can use them for training.

* Classic RNNs have memory issues (long-term dependencies). The beginning of the sequences that we use for training appears to be "forgotten" due to the overwhelming effect of more recent states.

* In general, these problems can be overcome by using gated RNNs. They can store information,just like having a memory, for later use. The data learns to read, write, and erase from the memory.

* The two most commonly used gated RNNs are Long Short-Term Memory Networks and Gated Recurrent Unit Neural Networks.We will try both these RNNs for or application and select one from it.


### Refer to stock_price_documentation.ipynb for further details on the implementation.

Output Samples:

*  **When 3 previous days prices as considered as features**

![graph1](/Data/Output_3days_features.png)


*  **When 5 previous days prices as considered as features**

![graph2](/Data/Output_5days_features.png)


If used give credits by forking, staring or watching git hub repo or in some other way.:slightly_smiling_face:
