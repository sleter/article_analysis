import tensorflow as tf
import pandas as pd

class Tensorflow_LSTM:
    def __init__(self, embed_size=300, max_word_len=50):
        print("Initializing ML submodule \n -----------------------------")
        self.embed_size = embed_size
        self.max_word_len = max_word_len
    
        self.df = pd.read_csv('Data/GatheredData/data_gathered_2019-09-03-2019-10-03_10437', index_col=0)
        self.df_embd = pd.read_csv('GoogleNewsModelData/EmbeddingsData/embeddings_2019-10-05_12:17:49.csv', index_col=0)
    
    def test_tensorflow(self):
        print("Tensorflow version: {}".format(tf.__version__))
        print("\nGPU available: {}".format(tf.test.is_gpu_available()))
        print("\nDevice name: {}\n".format(tf.random.uniform([3, 3]).device))
        # print("Benchmark config: {}\n".format(tf.test.benchmark_config()))
        




