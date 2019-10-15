import pandas as pd
import tensorflow as tf
import tensorflow as tf
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Simple_NN(AbstractNN):
    def __init__(self):
        super().__init__(str(__class__))
        
    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)
    
    def create_model(self):
        df = self.read_dataset("data_9415samples_2019-10-15_15:06:32")
        X, Y, X_width = self.split_dataset(df)
        model = Sequential()
        model.add(Dense(X_width, input_dim=X_width, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model)
        