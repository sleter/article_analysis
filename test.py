from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from datetime import date

def main():
    dh = DataHarvester()
    # df = dh.fetch_data(from_date=date(2019, 8, 1), to_date=date(2019, 8, 1))
    df = dh.fetch_top_articles()
    print(df.columns())
    
if __name__== "__main__":
  main()