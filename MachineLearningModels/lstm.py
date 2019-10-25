import tensorflow as tf
import pandas as pd
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class Tensorflow_LSTM(AbstractNN):
    def __init__(self, version, embed_size=300, max_word_len=50):
        super().__init__(str(__class__), version)
        self.embed_size = embed_size
        self.max_word_len = max_word_len
        
    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)

    def create_model(self, input_dim, voc_size, optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        # model.add(Dense(60, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Embedding(input_dim=voc_size, output_dim=64))
        
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer=init, activation=tf.nn.sigmoid))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])

        return model
    
    def fit_model(self):
        df = self.read_dataset("data_9415samples_2019-10-15_150632")
        X_train, X_test, y_train, y_test, X_width = self.split_dataset(df)
        
        voc_size = (X_train.max()+1).astype('int64')
        print(voc_size)

        model = KerasClassifier(build_fn=self.create_model, input_dim=X_width, voc_size=voc_size, verbose=1)

        model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test), verbose=1)
        loss, accuracy, auc = model.evaluate(X_test, y_test)
        print("loss: {} | accuracy: {} | auc: {}".format(loss, accuracy, auc))
        # self.save_model(model)
        # self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)




