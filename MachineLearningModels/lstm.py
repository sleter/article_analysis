import tensorflow as tf
import pandas as pd
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding, Bidirectional, concatenate
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from Preprocessing.data_preprocessing import DataPreprocessing

class Tensorflow_LSTM(AbstractNN):
    def __init__(self, version, embed_size=300, max_word_len=50):
        super().__init__(str(__class__), version)
        self.embed_size = embed_size
        self.max_word_len = max_word_len
        
    def read_dataset(self, filename="data_8502_lstm_samples_2019-10-27_14:30:28"):
        super().read_dataset(filename)
        df = pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)
        return df

    def create_model(self, meta_length, seq_length, vocab_size, optimizer='adam', init='glorot_uniform'):
        nlp_input = Input(shape=(seq_length,), name='nlp_input')
        meta_input = Input(shape=(meta_length,), name='meta_input')
        emd_layer = Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length)(nlp_input)
        nlp_output = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(emd_layer)
        join_nlp_meta = concatenate([nlp_output, meta_input])
        join_nlp_meta = Dense(120, activation='relu')(join_nlp_meta)
        join_nlp_meta = Dense(30, activation='relu')(join_nlp_meta)
        join_nlp_meta_output = Dense(1, activation='sigmoid')(join_nlp_meta)
        
        model = Model(inputs=[nlp_input, meta_input], outputs=[join_nlp_meta_output])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])

        return model


    def fit_model(self, save=False):
        df = self.read_dataset("data_8502_lstm_samples_2019-10-27_14:30:28")
        vocab_size=70227
        max_length=7
        X_train, X_test, X_train_meta, X_test_meta, y_train, y_test, X_width = self.split_dataset(df, astype=int, lstm=True, vocab_size=vocab_size, max_length=max_length)

        # model = KerasClassifier(build_fn=self.create_model, meta_length=X_width, seq_length=max_length, vocab_size=vocab_size, verbose=1)

        model = self.create_model(
            meta_length=X_width,
            seq_length=max_length,
            vocab_size=vocab_size
        )

        model.fit([X_train, X_train_meta], y_train, epochs=10, batch_size=100)

        loss, accuracy, auc = model.evaluate([X_test, X_test_meta], y_test)
        print("loss: {} | accuracy: {} | auc: {}".format(loss, accuracy, auc))

        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)




