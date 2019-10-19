import tensorflow as tf
import pandas as pd
from .nn_abc import AbstractNN

class Tensorflow_LSTM(AbstractNN):
    def __init__(self, embed_size=300, max_word_len=50):
        super().__init__(str(__class__))
        self.embed_size = embed_size
        self.max_word_len = max_word_len
    
    def read_dataset(self, filename):
        super().read_dataset(filename)
        




