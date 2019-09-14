from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from Preprocessing.data_preprocessing import DataPreprocessing
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
    dp = DataPreprocessing("GatheredData/data_gathered_2019-09-03-2019-09-12_5358")
    tokens = dp.create_tokens_from_title()
    dp.create_embeddings(tokens)

def main():
    preprocessing()
    # get_daily()
    # gather_all()
    
    
if __name__== "__main__":
  main()