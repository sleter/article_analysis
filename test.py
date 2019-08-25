from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from datetime import date

def main():
    dh = DataHarvester()
    dh.fetch_data(from_date=date(2019, 8, 17), to_date=date(2019, 8, 18))
  
if __name__== "__main__":
  main()