import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import tempfile
import json
import os
from utils.helpers import timing
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, Flatten
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score


class Simple_NN(AbstractNN):
    def __init__(self, version, filename):
        super().__init__(str(__class__), version, filename)
        
    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)

    def create_model(self, input_dim, optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(60, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(1, kernel_initializer=init, activation=tf.nn.sigmoid))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model

    def optimize_model(self):
        df = self.read_dataset(self.filename)

        X_train, X_test, y_train, y_test, X_width, (neg, pos, total) = self.split_dataset(df)

        model = KerasClassifier(build_fn=self.create_model, input_dim=X_width, verbose=1)

        #fisrt trial
        init = ['glorot_uniform']#, 'normal'] 
        #second trial
        # init = ['lecun_uniform', 'normal', 'zero', 'glorot_normal', 'he_normal', 'he_uniform']        

        optimizer = ['adam']#, 'rmsprop']
        batch_sizes = [10]#, 20, 50]
        epochs = [10, 15]

        param_grid = dict(epochs=epochs, batch_size=batch_sizes, init=init, optimizer=optimizer)
        gscv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        gscv_result = gscv.fit(X_train, y_train)

        results = {
            'Best accuracy='+str(gscv_result.best_score_): gscv_result.best_params_
        }
        accs = gscv_result.cv_results_['mean_test_score']
        stds = gscv_result.cv_results_['std_test_score']
        params = gscv_result.cv_results_['params']
        # print(gscv_result.cv_results_.keys())

        for acc, stdev, param in zip(accs, stds, params):
            results['Accuracy='+str(acc)+'|Stdev='+str(stdev)] = param

        with open('MachineLearningModels/OptimizationResults/{model_name}_{date:%Y-%m-%d}_{version}.json'.format(model_name = self.name, date = datetime.datetime.now(), version = self.version), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f'Best params: {gscv_result.best_params_}')
        return gscv_result.best_params_, X_train, X_test, y_train, y_test, X_width, (neg, pos, total)

    @timing
    def fit_optimize_eval_model(self, save=False):
        best_params, X_train, X_test, y_train, y_test, X_width, (neg, pos, total) = self.optimize_model()
        
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']
        init = best_params['init']
        optimizer = best_params['optimizer']

        model = self.create_model(
            X_width,
            optimizer=optimizer,
            init=init)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
        print("\n\nEvaluation on test set\n")
        print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))
        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)

    @timing
    def fit_model(self, epochs, batch_size, init, optimizer, save=False):
        df = self.read_dataset(self.filename)

        X_train, X_test, y_train, y_test, X_width, (neg, pos, total) = self.split_dataset(df)
        model = self.create_model(
            X_width,
            optimizer=optimizer,
            init=init)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        self.plot_metrics(history, val=False, meta_text='snn')

        loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
        print("\n\nEvaluation on test set\n")
        print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))
        if save:
            filepath = self.save_model(model, return_name=True)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc, precision=precision, recall=recall)


class Complex_NN(AbstractNN):
    def __init__(self, version, filename):
        super().__init__(str(__class__), version, filename)

    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)

    def create_model(self, input_dim, optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(60, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.3)),
        model.add(Dense(120, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.3))
        model.add(Dense(100, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.3))
        model.add(Dense(30, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(1, kernel_initializer=init, activation=tf.nn.sigmoid))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    @timing
    def fit_model(self, epochs, batch_size, init, optimizer, save=False):
        df = self.read_dataset(self.filename)

        X_train, X_test, y_train, y_test, X_width, (neg, pos, total) = self.split_dataset(df)

        weight_for_0 = (1 / neg)*(total)/2.0
        weight_for_1 = (1 / pos)*(total)/2.0

        class_weight = {0: weight_for_0, 1: weight_for_1}

        model = self.create_model(
            X_width,
            optimizer=optimizer,
            init=init,
            class_weight=class_weight)
        hisotry = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        self.plot_metrics(hisotry, val=False, meta_text='cnn')

        loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
        print("\n\nEvaluation on test set\n")
        print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))
        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc, precision=precision, recall=recall)

class Complex_NN_title(AbstractNN):
    def __init__(self, version, filename):
        super().__init__(str(__class__), version, filename)
        self.vocab_size = 70227
        self.max_length = 7

    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)

    def create_model(self, input_dim, seq_length, vocab_size, optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        # model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_length))
        # model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)))),
        model.add(Dense(60, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.3)),
        model.add(Dense(120, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.3))
        model.add(Dense(100, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dropout(0.3))
        model.add(Dense(30, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Flatten())
        model.add(Dense(1, kernel_initializer=init, activation=tf.nn.sigmoid))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    @timing
    def fit_model(self, epochs, batch_size, init, optimizer, save=False):
        df = self.read_dataset(self.filename)

        new_columns = [i for i in df.columns if i.startswith('title_embedding') or i.startswith('content_embedding') or i == "top_article"]
        df = df[new_columns]

        # X_width = 2
        # X_train, X_test, X_val, _, X_test_meta, _, y_train, y_test, y_val, _, (neg, pos, total) = self.split_dataset(df, astype=int, lstm=True, vocab_size=self.vocab_size, max_length=self.max_length)
        X_train, X_test, y_train, y_test, X_width, (neg, pos, total) = self.split_dataset(df)

        weight_for_0 = (1 / neg)*(total)/2.0
        weight_for_1 = (1 / pos)*(total)/2.0

        class_weight = {0: weight_for_0, 1: weight_for_1}

        model = self.create_model(
            input_dim = X_width,
            seq_length=self.max_length,
            vocab_size=self.vocab_size,
            optimizer=optimizer,
            init=init)

        prehistory = model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=batch_size,
            # validation_data = (X_val, y_val),
            class_weight=class_weight)

        # Saving model weights | keeping these in temp file
        initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
        model.save_weights(initial_weights)

        model = self.create_model(
            input_dim = X_width,
            seq_length=self.max_length,
            vocab_size=self.vocab_size,
            optimizer=optimizer,
            init=init)
        model.load_weights(initial_weights)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            # validation_data = (X_val, y_val),
            class_weight=class_weight)

        # print(history.history.keys())
        # self.plot_metrics(prehistory, meta_text="cnn_pre")
        # self.plot_metrics(history, meta_text="cnn")

        loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
        print("\n\nEvaluation on test set\n")
        print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))  

        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc, precision=precision, recall=recall)
