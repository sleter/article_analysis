from utils.read_config import ConfigReader
from utils.helpers import flatten, get_dates_between
from newsapi import NewsApiClient
from .sharedcount import SharedCountApiClient
from multiprocessing.pool import Pool
import socialshares
import datetime
import pandas as pd
import json
import progressbar

class DataHarvester():
    def __init__(self):
        cf = ConfigReader('config.json')
        self.newsapi = NewsApiClient(api_key=cf.get_news_api_key())
        self.sharedcount = SharedCountApiClient(api_key=cf.get_sharedcount_api_key())
        
        self.sources = \
            ["abc-news"]
            # "al-jazeera-english",
            # "business-insider",
            # "cbs-news",
            # "cnn",
            # "google-news",
            # "nbc-news",
            # "newsweek",
            # "reddit-r-all",
            # "the-irish-times",
            # "the-new-york-times",
            # "reuters",
            # "the-wall-street-journal",
            # "espn",
            # "bbc-news"]
        
        # self.widgets=[
        # ' [', progressbar.Timer(), '] ',
        # progressbar.Bar(),
        # ' (', progressbar.ETA(), ') ',
        # ]
        # for i in progressbar.progressbar(range(1, 1), redirect_stdout=True, widgets=self.widgets):
    
    def get_en_news_sources(self):
        all_sources = self.newsapi.get_sources(language="en")
        sources_list = []
        for key, value in all_sources.items():
            if key == "sources":
                for s in value:
                    sources_list.append(s["id"])
        return sources_list
    
    def _create_df(self, articles):
        flattened_articles = [flatten(article) for article in articles]
        return pd.DataFrame(flattened_articles)
    
    def fetch_data_daily(self, date, language, sort_by, page_size):
        df = pd.DataFrame()
        for source in self.sources:
            data_everything = self.newsapi.get_everything(sources=source, from_param=str(date), to=str(date), language=language, sort_by=sort_by, page_size=page_size)
            df_temp = self._create_df(data_everything["articles"])
            df = df.append(df_temp)
        return df
    
    def fetch_data(self, from_date=datetime.date.today(), to_date=datetime.date.today(), language='en', sort_by=None, page_size=100):
        dates = get_dates_between(from_date, to_date)
        df = pd.DataFrame()
        
        pool = Pool(processes=len(dates))
        input_parameters = [(date, language, sort_by, page_size) for date in dates]
        dfs = pool.starmap(self.fetch_data_daily, input_parameters)
        for df_elem in dfs:
            df = df.append(df_elem) 
        pool.close()
        return df
    
    def fetch_top_articles(self):
        df = pd.DataFrame()
        for source in self.sources:
            data_top = self.newsapi.get_top_headlines(q=None, sources=source, language='en', country=None, category=None, page_size=100)
            df_temp = self._create_df(data_top["articles"])
            df = df.append(df_temp)
        return df
    
    def append_top_articles(self, df):
        pass 
    
    def save_csv(self, df, path):
        df.to_csv(path)