from abc import ABC, abstractmethod
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import ast, re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy, datetime, re, json

class AbstractNN(ABC):
    @abstractmethod
    def __init__(self, submodule_name, version, filename):
        # print(__class__)
        print("Initializing ML submodule: {} \n -----------------------------".format(submodule_name))
        numpy.random.seed(42) # answer to the ultimate question of life, universe and everything
        self.name = submodule_name[33:-2].lower()
        self.version = version
        self.filename = filename
        mpl.rcParams['figure.figsize'] = (12, 10)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    @abstractmethod
    def read_dataset(self, filename):
        print("Reading {} dataset \n -----------------------------".format(filename))

    def save_model(self, model):
        model.save('MachineLearningModels/SavedModels/{model_name}_{date:%Y-%m-%d}_{version}.h5'.format( \
            model_name = self.name, \
            date = datetime.datetime.now(),\
            version = self.version))

    def save_metadata(self, **kwargs):
        with open('MachineLearningModels/SavedModels/metadata_{model_name}_{date:%Y-%m-%d}_{version}.json'.format(model_name = self.name, date = datetime.datetime.now(), version = self.version), 'w', encoding='utf-8') as f:
            json.dump(str(kwargs), f, ensure_ascii=False, indent=4)
        
    def split_dataset(self, df, Y_name = "top_article", balance=True, astype = float, lstm = False, vocab_size = 70227, max_length = 7):
        neg, pos = np.bincount(df[Y_name])
        total = neg + pos
        if lstm:
            df['title'] = df['title'].astype(str)
            dataset_X = df.loc[:, df.columns != Y_name].values
            dataset_Y = df[Y_name].values
            X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

            X_train_meta = np.delete(X_train, 1, 1)
            X_test_meta = np.delete(X_test, 1, 1)
            X_val_meta = np.delete(X_val, 1, 1)

            X_train_nlp = X_train[:, [1]]
            X_test_nlp = X_test[:, [1]]
            X_val_nlp = X_val[:, [1]]

            encoded_titles_X_train = []
            encoded_titles_X_test = []
            encoded_titles_X_val = []

            for title in X_train_nlp:
                encoded_titles_X_train.append(one_hot(title[0], vocab_size))
            for title in X_test_nlp:
                encoded_titles_X_test.append(one_hot(title[0], vocab_size))
            for title in X_val_nlp:
                encoded_titles_X_val.append(one_hot(title[0], vocab_size))

            padded_titles_X_train = pad_sequences(encoded_titles_X_train, maxlen=max_length, padding='post')
            padded_titles_X_test = pad_sequences(encoded_titles_X_test, maxlen=max_length, padding='post')
            padded_titles_X_val = pad_sequences(encoded_titles_X_val, maxlen=max_length, padding='post')

            print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

            return padded_titles_X_train, padded_titles_X_test, padded_titles_X_val, X_train_meta, X_test_meta, X_val_meta, y_train, y_test, y_val, X_train_meta.shape[1], (neg, pos, total)
        else:
            dataset_X = df.loc[:, df.columns != Y_name].values
            dataset_Y = df[Y_name].values
            X = dataset_X.astype(astype)

            X_train, X_test, y_train, y_test = train_test_split(X, dataset_Y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test, dataset_X.shape[1], (neg, pos, total)
    
    def plot_metrics(self, history, val=True, meta_text=""):
        if meta_text == "lstm" or meta_text == "cnn":
            metrics =  ['loss', 'auc_1', 'precision_1', 'recall_1']
        else:
            metrics =  ['loss', 'auc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch,  history.history[metric], color=self.colors[0], label='Train')
            if val:
                plt.plot(history.epoch, history.history["val_"+metric],color=self.colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc' or 'auc_1':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])
            plt.legend()
        plt.savefig('MachineLearningModels/ModelsMetricsPlots/metrics_plot_{}_{date:%Y-%m-%d}.png'.format(meta_text, date = datetime.datetime.now()), bbox_inches='tight')


    def test_tensorflow(self):
        print("Tensorflow version: {}".format(tf.__version__))
        print("\nGPU available: {}".format(tf.test.is_gpu_available()))
        print("\nDevice name: {}\n".format(tf.random.uniform([3, 3]).device))
        # print("Benchmark config: {}\n".format(tf.test.benchmark_config()))
