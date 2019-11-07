# from DataHarvesting.data_harvester import DataHarvester
# from DataHarvesting.sharedcount import SharedCountApiClient
# from Preprocessing.data_preprocessing import DataPreprocessing
# from MachineLearningModels.lstm import Tensorflow_LSTM
from MachineLearningModels.nn import Simple_NN, Complex_NN
# from DataAnalysis.data_analyzer import DataAnalyzer
from datetime import date
import pandas as pd
from utils.helpers import timing

# def preprocessing():
#     dp = DataPreprocessing("GatheredData/data_gathered_2019-09-03-2019-11-04_23535")
#     # dp.save_embeddings(columns=True, word=False)
#     # dp.generate_wordcloud()
#     # dp.tsne_dim_red()
#     dp.create_samples(filename="categorical_embeddings_2019-11-05_16:06:37", embeddings=True)
#     # dp.create_samples(filename="data_gathered_2019-09-03-2019-11-04_23535", embeddings=False)

# def analyze_data():
#     da = DataAnalyzer()
#     # da.show_correlation_matrix(da.samples_df)
#     da.print_columns_as_list(da.samples_df)

def main():
    # preprocessing()
    # analyze_data()
    pass
    
    
if __name__== "__main__":
  main()