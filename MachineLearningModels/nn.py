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

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def optimize_model(self):
        df = self.read_dataset("data_9415samples_2019-10-15_150632")
        X_train, X_test, y_train, y_test, X_width = self.split_dataset(df)

        model = KerasClassifier(build_fn=self.create_model, input_dim=X_width, verbose=1)

        optimizer = ['adam', 'sgd', 'rmsprop']
        init = ['glorot_uniform', 'uniform'] 
        batch_sizes = [64, 128, 512]
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

        # model.fit(X_train, y_train, epochs=100, batch_size=100)
        # loss, accuracy = model.evaluate(X_test, y_test)
        # print("loss: {} | accuracy: {}".format(loss, accuracy))
        # self.save_model(model)
        # self.save_metadata(loss = loss, accuracy = accuracy)
        