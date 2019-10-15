from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from Preprocessing.data_preprocessing import DataPreprocessing
from MachineLearningModels.lstm import Tensorflow_LSTM
from MachineLearningModels.nn import Simple_NN
from DataAnalysis.data_analyzer import DataAnalyzer
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
    # dp.save_embeddings(columns=True, word=False)
    # dp.generate_wordcloud()
    # dp.tsne_dim_red()
    dp.create_samples(embeddings_filename="categorical_embeddings_2019-10-15_15:03:14")

def ml_stuff():
    nn = Simple_NN()
    nn.create_model()

def analyze_data():
    da = DataAnalyzer()
    # da.show_correlation_matrix(da.samples_df)
    da.print_columns_as_list(da.samples_df)

def main():
    # preprocessing()
    # get_daily()
    # gather_all()
    ml_stuff()
    # analyze_data()
    
    
if __name__== "__main__":
  main()