from abc import ABC, abstractmethod
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy, datetime, re, json

class AbstractNN(ABC):
    @abstractmethod
    def __init__(self, submodule_name, version):
        # print(__class__)
        print("Initializing ML submodule: {} \n -----------------------------".format(submodule_name))
        numpy.random.seed(42) # answer to the ultimate question of life, universe and everything
        self.name = submodule_name[33:-2].lower()
        self.version = version
        
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
        
    def split_dataset(self, df, Y_name="top_article"):    
        dataset_X = df.loc[:, df.columns != Y_name].values
        dataset_Y = df[Y_name].values
        X = dataset_X.astype(float)
        Y = dataset_Y
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, dataset_X.shape[1]
    
    def test_tensorflow(self):
        print("Tensorflow version: {}".format(tf.__version__))
        print("\nGPU available: {}".format(tf.test.is_gpu_available()))
        print("\nDevice name: {}\n".format(tf.random.uniform([3, 3]).device))
        # print("Benchmark config: {}\n".format(tf.test.benchmark_config()))
