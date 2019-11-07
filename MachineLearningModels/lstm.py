import tensorflow as tf
import pandas as pd
import json, datetime
from utils.helpers import timing
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding, Bidirectional, concatenate
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from Preprocessing.data_preprocessing import DataPreprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


class Tensorflow_LSTM(AbstractNN):
    def __init__(self, version, filename, embed_size=300, max_word_len=50):
        self.vocab_size = 70227
        self.max_length = 7
        super().__init__(str(__class__), version, filename)
        
    def read_dataset(self, filename="data_8502_lstm_samples_2019-10-27"):
        super().read_dataset(filename)
        df = pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)
        return df

    def create_model(self, meta_length, seq_length, vocab_size, optimizer='adam', init='glorot_uniform'):
        nlp_input = Input(shape=(seq_length,), name='nlp_input')
        meta_input = Input(shape=(meta_length,), name='meta_input')
        emd_layer = Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length)(nlp_input)
        nlp_output = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(emd_layer)
        join_nlp_meta = concatenate([nlp_output, meta_input])
        join_nlp_meta = Dense(120, activation='relu')(join_nlp_meta)
        join_nlp_meta = Dense(30, activation='relu')(join_nlp_meta)
        join_nlp_meta_output = Dense(1, activation='sigmoid')(join_nlp_meta)
        
        model = Model(inputs=[nlp_input, meta_input], outputs=[join_nlp_meta_output])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])

        return model

    def optimize_model(self):
        df = self.read_dataset(self.filename)
        X_train, X_test, X_train_meta, X_test_meta, y_train, y_test, X_width = self.split_dataset(df, astype=int, lstm=True, vocab_size=self.vocab_size, max_length=self.max_length)

        model = KerasClassifier(build_fn=self.create_model, meta_length=X_width, seq_length=self.max_length, vocab_size=self.vocab_size, verbose=1)

        optimizer = ['adam', 'rmsprop']
        init = ['glorot_uniform', 'uniform'] 
        batch_sizes = [10, 20, 30]
        epochs = [10, 20, 40]

        param_grid = dict(epochs=epochs, batch_size=batch_sizes, init=init, optimizer=optimizer)
        gscv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        # GridSearchCV cannot accept multiple inputs !!!
        gscv_result = gscv.fit([X_train, X_train_meta], y_train)

        results = {
            'Best accuracy='+str(gscv_result.best_score_): gscv_result.best_params_
        }
        accs = gscv_result.cv_results_['mean_test_score']
        stds = gscv_result.cv_results_['std_test_score']
        params = gscv_result.cv_results_['params']

        for acc, stdev, param in zip(accs, stds, params):
            results['Accuracy='+str(acc)+'|Stdev='+str(stdev)] = param

        with open('MachineLearningModels/OptimizationResults/{model_name}_{date:%Y-%m-%d}_{version}.json'.format(model_name = self.name, date = datetime.datetime.now(), version = self.version), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f'Best params: {gscv_result.best_params_}')
        return gscv_result.best_params_, X_train, X_test, X_train_meta, X_test_meta, y_train, y_test, X_width

    def fit_optimize_eval_model(self, save=True):
        best_params, X_train, X_test, X_train_meta, X_test_meta, y_train, y_test, X_width = self.optimize_model()
        
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']
        init = best_params['init']
        optimizer = best_params['optimizer']

        model = self.create_model(
            meta_length=X_width,
            seq_length=self.max_length,
            vocab_size=self.vocab_size,
            optimizer=optimizer,
            init=init
        )

        model.fit([X_train, X_train_meta], y_train, epochs=epochs, batch_size=batch_size)
        loss, accuracy, auc = model.evaluate([X_test, X_test_meta], y_test)
        print("loss: {} | accuracy: {} | auc: {}".format(loss, accuracy, auc))
        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)

    @timing
    def fit_model(self, save=False, epochs = 20, batch_size = 20, optimizer = 'adam',init = 'glorot_uniform'):
        df = self.read_dataset(self.filename)
        X_train, X_test, X_train_meta, X_test_meta, y_train, y_test, X_width = self.split_dataset(df, astype=int, lstm=True, vocab_size=self.vocab_size, max_length=self.max_length)

        model = self.create_model(
            meta_length=X_width,
            seq_length=self.max_length,
            vocab_size=self.vocab_size,
            optimizer=optimizer,
            init=init
        )

        model.fit([X_train, X_train_meta], y_train, epochs=epochs, batch_size=batch_size)

        loss, accuracy, auc = model.evaluate([X_test, X_test_meta], y_test)
        print("loss: {} | accuracy: {} | auc: {}".format(loss, accuracy, auc))

        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)




