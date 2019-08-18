from utils.read_config import ConfigReader
from utils.helpers import flatten
from newsapi import NewsApiClient
from .sharedcount import SharedCountApiClient
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
        # self.widgets=[
        # ' [', progressbar.Timer(), '] ',
        # progressbar.Bar(),
        # ' (', progressbar.ETA(), ') ',
        # ]
        # for i in progressbar.progressbar(range(1, 1), redirect_stdout=True, widgets=self.widgets):
    
    def get_en_news_sources(self):
        sources = self.newsapi.get_sources(language="en")
        sources_list = []
        for key, value in sources.items():
            if key == "sources":
                for s in value:
                    sources_list.append(s["id"])
        return sources_list
    
    def _create_df(self, articles):
        flattened_articles = [flatten(article) for article in articles]
        return pd.DataFrame(flattened_articles)
    
    def fetch_data(self, from_date=datetime.datetime.now().date(), to_date=datetime.datetime.now().isoformat(), language='en', sort_by=None, page_size=100):
        sources = "abc-news," + \
                  "al-jazeera-english," + \
                  "business-insider," + \
                  "cbs-news," + \
                  "cnn," + \
                  "google-news," + \
                  "nbc-news," + \
                  "newsweek," + \
                  "reddit-r-all," + \
                  "the-irish-times," + \
                  "the-new-york-times," + \
                  "reuters," + \
                  "the-wall-street-journal," + \
                  "espn," + \
                  "bbc-news"
        
        data_everything = self.newsapi.get_everything(sources=sources, from_param=str(from_date), to=to_date, language=language, sort_by=sort_by, page_size=page_size)
        df_everything = self._create_df(data_everything["articles"])
        print(df_everything)
        
        