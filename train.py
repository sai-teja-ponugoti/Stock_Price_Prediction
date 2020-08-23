import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import LSTM,GRU
from keras.layers import Dropout
from keras.layers import Dense


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


# function to create train and test data sets form the given original dataset
def create_train_test_data():
    # reading the original csv file
    data = pd.read_csv("./data/stock_data.csv")

    # ignoring the two unnecessary columns of the dataset
    # ignoring Date and Close/Last columns as they dont constitue anything to our problem
    data = data[data.columns[[2, 3, 4, 5]]]

    shift_window = -3

    dataframe = pd.concat([data.shift(shift_window), data.shift(shift_window + 1), data.shift(shift_window + 2), data],
                          axis=1)
    dataframe.columns = ['v3', 'o3', 'h3', 'l3', 'v2', 'o2', 'h2', 'l2', 'v1', 'o1',
                         'h1', 'l1', 'v0', 'label', 'h0', 'l0']

    # ignoring last 3 rows of the result processed data as they are not valid andhas Nan in them
    dataframe = dataframe.iloc[:dataframe.shape[0] + shift_window]

    # inoring extra colums volume,high,low of the last set as they are next days unwanted columns
    dataframe = dataframe[['v3', 'o3', 'h3', 'l3', 'v2', 'o2', 'h2', 'l2', 'v1', 'o1',
                           'h1', 'l1', 'label', ]]

    dataframe['index'] = [-i for i in range(dataframe.shape[0])]
    # shuffling the data using sklearn shuffle function
    dataframe = shuffle(dataframe)

    # splitting data to train and test parts
    # using random state so that the results can be replicated
    train_data, test_data = train_test_split(dataframe, test_size=0.3, random_state=100)

    # storing the split data into respective files
    train_data.to_csv('./data/train_data_RNN.csv', index=False)
    test_data.to_csv('./data/test_data_RNN.csv', index=False)

    print("train and test data set creation is finished")


def load_train_data():
    train_data = pd.read_csv("./data/train_data_RNN.csv")
    train_data = np.array(train_data)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_data[:, 0:13])
    train_data = scaler.transform(train_data[:, 0:13])

    x_train = train_data[:, 0:12]
    x_train = np.reshape(x_train, (879, 3, 4))
    y_train = train_data[:, 12]
    y_train = np.asarray(y_train)

    print("finished loading training data")
    return x_train, y_train


# function to create the model
def create_model(input_shape_length):
    model = Sequential()

    # model.add(GRU(units=50, return_sequences=True, input_shape=(input_shape_length, 4)))
    model.add(GRU(units=50, input_shape=(input_shape_length, 4)))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(GRU(units=50))
    # model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    print("model creation is done")
    return model


if __name__ == "__main__":
    # 1. load your training data
    create_train_test_data()
    # uncomment the above function to create train and test data sets
    x_train, y_train = load_train_data()

    # 2. Train your network
    model = create_model(x_train.shape[1])
    print("Starting to train the model")
    history = model.fit(x_train, y_train, epochs=100, batch_size=32)
    # Make sure to print your training loss within training to show progress
    # Make sure you print the final training loss
    print("final training loss: ", history.history['loss'][-1])
    print("finished training mode and saving the model")

    # 3. Save your model
    model.save("./models/RNN_model.h5")
