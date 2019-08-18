from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient

def main():
    dh = DataHarvester()
    dh.fetch_data(from_date='2019-08-17', to_date='2019-08-17')
  
if __name__== "__main__":
  main()