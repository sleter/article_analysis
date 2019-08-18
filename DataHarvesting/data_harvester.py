from utils.read_config import ConfigReader
from newsapi import NewsApiClient
from .sharedcount import SharedApiCountClient
import socialshares

class DataHarvester():
    def __init__(self):
        cf = ConfigReader('config.json')
        self.newsapi = NewsApiClient(api_key=cf.get_news_api_key())
        self.sharedcount = SharedApiCountClient(api_key=cf.get_sharedcount_api_key())
    
    def get_en_news_sources(self):
        return self.newsapi.get_sources(language="en")
    
    
    def fetch_data(self):
        pass
        
        