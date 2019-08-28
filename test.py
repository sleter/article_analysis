from DataHarvesting.data_harvester import DataHarvester
from DataHarvesting.sharedcount import SharedCountApiClient
from datetime import date
import pandas as pd
from utils.helpers import timing

def main():
    dh = DataHarvester()
    # df1 = dh.fetch_data(from_date=date(2019, 8, 25), to_date=date(2019, 8, 26))
    # df2 = dh.fetch_top_articles()
    # df1.to_csv("Data/fetch_data.csv")
    
    # df1 = pd.read_csv("Data/fetch_data.csv", index_col=0)
    # df2 = pd.read_csv("Data/fetch_top.csv", index_col=0)
    # df = dh.append_top_articles(df1, df2)
    # df.to_csv("Data/data_everything_with_top.csv")
    
    # df = pd.read_csv("Data/data_everything_with_top.csv")
    # print(dh.change_column_names(df).columns)
    
    # dh.append_social_shares(df1)
    
    # dh.append_facebook_engagement(df1).to_csv("Data/data_all_test.csv")
    
    time = dh.harvest_daily()
    print("Process took: {}".format(time)+' seconds\n\n')
    input("Press Enter to continue...")
    
    
if __name__== "__main__":
  main()