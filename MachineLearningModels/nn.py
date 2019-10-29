import pandas as pd
import tensorflow as tf
import datetime
import json
from .nn_abc import AbstractNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


class Simple_NN(AbstractNN):
    def __init__(self, version):
        super().__init__(str(__class__), version)
        
    def read_dataset(self, filename):
        super().read_dataset(filename)
        return pd.read_csv("Data/PreprocessedData/{}.csv".format(filename), index_col=0)

    def create_model(self, input_dim, optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(60, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(1, kernel_initializer=init, activation=tf.nn.sigmoid))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])

        return model

    def optimize_model(self):
        df = self.read_dataset("data_9415samples_2019-10-15_150632")
        X_train, X_test, y_train, y_test, X_width = self.split_dataset(df)

        model = KerasClassifier(build_fn=self.create_model, input_dim=X_width, verbose=1)

        #fisrt trial
        # init = ['glorot_uniform', 'uniform'] 
        #second trial
        init = ['lecun_uniform', 'normal', 'zero', 'glorot_normal', 'he_normal', 'he_uniform']        

        optimizer = ['adam', 'sgd', 'rmsprop']
        batch_sizes = [20, 50, 100, 300]
        epochs = [10, 20, 50, 100]

        param_grid = dict(epochs=epochs, batch_size=batch_sizes, init=init, optimizer=optimizer)
        gscv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        gscv_result = gscv.fit(X_train, y_train)

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
        return gscv_result.best_params_, X_train, X_test, y_train, y_test, X_width

    def fit_optimize_eval_model(self):
        best_params, X_train, X_test, y_train, y_test, X_width = self.optimize_model()
        
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']
        init = best_params['init']
        optimizer = best_params['optimizer']

        model = self.create_model(
            X_width,
            optimizer=optimizer,
            init=init)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        loss, accuracy, auc = model.evaluate(X_test, y_test)
        print("loss: {} | accuracy: {} | auc: {}".format(loss, accuracy, auc))
        self.save_model(model)
        self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)

    def fit_model(self, epochs, batch_size, init, optimizer, save=False):
        df = self.read_dataset("data_9415samples_2019-10-15_150632")
        X_train, X_test, y_train, y_test, X_width = self.split_dataset(df)
        model = self.create_model(
            X_width,
            optimizer=optimizer,
            init=init)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        loss, accuracy, auc = model.evaluate(X_test, y_test)
        print("loss: {} | accuracy: {} | auc: {}".format(loss, accuracy, auc))
        if save:
            self.save_model(model)
            self.save_metadata(loss = loss, accuracy = accuracy, auc=auc)


class Complex_NN(Simple_NN):
    def __init__(self, version):
        AbstractNN.__init__(self, submodule_name=str(__class__), version=version)

    def create_model(self, input_dim, optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(60, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(120, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(30, input_dim=input_dim, kernel_initializer=init, activation=tf.nn.relu))
        model.add(Dense(1, kernel_initializer=init, activation=tf.nn.sigmoid))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])
        return model