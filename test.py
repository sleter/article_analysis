from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from datetime import date
import pandas as pd
from utils.helpers import timing

def main():
    dh = DataHarvester()    
    harvest_time = dh.harvest_daily()
    print("Process took: {}".format(harvest_time)+' seconds\n\n')
    input("Press Enter to continue...")
    
if __name__== "__main__":
  main()