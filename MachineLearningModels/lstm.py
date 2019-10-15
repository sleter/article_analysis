import tensorflow as tf
import pandas as pd
from .nn_abc import AbstractNN

class Tensorflow_LSTM(AbstractNN):
    def __init__(self, embed_size=300, max_word_len=50):
        super().__init__(str(__class__))
        self.embed_size = embed_size
        self.max_word_len = max_word_len
    
    def test_tensorflow(self):
        print("Tensorflow version: {}".format(tf.__version__))
        print("\nGPU available: {}".format(tf.test.is_gpu_available()))
        print("\nDevice name: {}\n".format(tf.random.uniform([3, 3]).device))
        # print("Benchmark config: {}\n".format(tf.test.benchmark_config()))
    
    def read_dataset(self, filename):
        super().read_dataset(filename)
        




