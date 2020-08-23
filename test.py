# import required packages
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
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def load_test_data():
    train_data = pd.read_csv("./data/train_data_RNN.csv")
    train_data = np.array(train_data)
    test_data = pd.read_csv("./data/test_data_RNN.csv")
    test_data = np.array(train_data)
    # arranging the test dataset in order of dates , so that graph will not be clumsy
    test_data = test_data[np.argsort(test_data[:, 13])]
    # test_data = test_data[:,0:13]

    scaler_min_max = MinMaxScaler(feature_range=(0, 1)).fit(train_data[:, 0:13])
    train_data = scaler_min_max.transform(train_data[:, 0:13])
    test_data = scaler_min_max.transform(test_data[:, 0:13])

    x_test = test_data[:, 0:12]
    x_test = np.reshape(x_test, (879, 3, 4))
    y_test = test_data[:, 12]
    y_test = np.asarray(y_test)

    return x_test, y_test, scaler_min_max


if __name__ == "__main__":
    # 1. Load your saved model
    model = tf.keras.models.load_model("./models/RNN_model.h5")

    # 2. Load your testing data
    x_test, y_test, scaler_min_max = load_test_data()

    # 3. Run prediction on the test data and output required plot and loss
    # print(type(y_test))
    # print(predicted_stock_price)
    predicted_stock_price = model.predict(x_test)
    # predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    y_test = (y_test * scaler_min_max.data_range_[12] + scaler_min_max.data_min_[12])
    predicted_stock_price = (predicted_stock_price * scaler_min_max.data_range_[12] + scaler_min_max.data_min_[12])

    print("testing Mean Squared Loss :",mean_squared_error(y_test, predicted_stock_price))

    plt.figure(figsize=(12, 12))
    plt.plot(y_test, color='black', label='original Stock Price')
    plt.plot(predicted_stock_price, color='orange', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time(old to present(left to right))')
    plt.xticks([])
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
