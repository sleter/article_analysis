from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from Preprocessing.data_preprocessing import DataPreprocessing
from MachineLearningModels.lstm import Tensorflow_LSTM
from datetime import date
import pandas as pd
from utils.helpers import timing

def get_daily():
    dh = DataHarvester()    
    harvest_time = dh.harvest_daily()
    # harvest_time = dh.harvest_top_daily()
    print("Process took: {}".format(harvest_time)+' seconds\n\n')
    input("Press Enter to continue...")

def gather_all():
    dh = DataHarvester()
    gathering_time = dh.gather_all()   
    print("Process took: {}".format(gathering_time)+' seconds\n\n')
    input("Press Enter to continue...")

def preprocessing():
    dp = DataPreprocessing("GatheredData/data_gathered_2019-09-03-2019-10-03_10437")
    dp.save_embeddings(title=True, word=False)
    # dp.generate_wordcloud()
    # dp.tsne_dim_red()

def ml_stuff():
    tlstm = Tensorflow_LSTM()
    tlstm.test_tensorflow()

def main():
    # preprocessing()
    get_daily()
    # gather_all()
    # ml_stuff()
    
    
if __name__== "__main__":
  main()