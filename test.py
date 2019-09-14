from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
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

def main():
    get_daily()
    # gather_all()
    
    
if __name__== "__main__":
  main()