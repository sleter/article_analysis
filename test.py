from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient

def main():
    dh = DataHarvester()
    print(dh.get_en_news_sources)
  
if __name__== "__main__":
  main()