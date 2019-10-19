from abc import ABC, abstractmethod
import tensorflow as tf

class AbstractNN(ABC):
    @abstractmethod
    def __init__(self, submodule_name):
        # print(__class__)
        print("Initializing ML submodule: {} \n -----------------------------".format(submodule_name))
        
    @abstractmethod
    def read_dataset(self, filename):
        print("Reading {} dataset \n -----------------------------".format(filename))
        
    def split_dataset(self, df, Y_name="top_article"):    
        dataset_X = df.loc[:, df.columns != Y_name].values
        dataset_Y = df[Y_name].values
        X = dataset_X.astype(float)
        Y = dataset_Y
        return X, Y, dataset_X.shape[1]
    
    def test_tensorflow(self):
        print("Tensorflow version: {}".format(tf.__version__))
        print("\nGPU available: {}".format(tf.test.is_gpu_available()))
        print("\nDevice name: {}\n".format(tf.random.uniform([3, 3]).device))
        # print("Benchmark config: {}\n".format(tf.test.benchmark_config()))
