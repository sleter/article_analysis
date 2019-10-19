import pandas as pd
import tensorflow as tf
import datetime
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

class Simple_NN(AbstractNN):
    def __init__(self):
        super().__init__(str(__class__))
        
    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)
    
    def create_model(self):
        df = self.read_dataset("test_linux")
        X_train, X_test, y_train, y_test, X_width = self.split_dataset(df)
        
        model = Sequential()
        model.add(Dense(X_width, input_dim=X_width, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=100)
        loss, accuracy = model.evaluate(X_test, y_test)
        print("loss: {} | accuracy: {}".format(loss, accuracy))
        model.save('MachineLearningModels/SavedModels/snn_model_{date:%Y-%m-%d_%H:%M:%S}.h5'.format(date=datetime.datetime.now()))