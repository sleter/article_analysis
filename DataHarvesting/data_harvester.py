from utils.read_config import ConfigReader
from utils.helpers import flatten, get_dates_between, timing
from newsapi import NewsApiClient
from .sharedcount import SharedCountApiClient
from multiprocessing.pool import Pool
import socialshares
import datetime
import pandas as pd
import json
import time
import progressbar

class DataHarvester():
    def __init__(self):
        cf = ConfigReader('config.json')
        self.newsapi = NewsApiClient(api_key=cf.get_news_api_key())
        self.fb_api_keys = cf.get_facebookgraph_api_key()
        self.sharedcount = SharedCountApiClient(social_share_api_key=cf.get_sharedcount_api_key(),\
                                                facebook_graph_api_key=self.fb_api_keys[0])
        self.sources = \
            ["abc-news",
            "al-jazeera-english",
            "business-insider",
            "cbs-news",
            "cnn",
            # "google-news", # wrong urls
            "nbc-news",
            "newsweek",
            "reddit-r-all",
            "the-irish-times",
            "the-new-york-times",
            "reuters",
            "the-wall-street-journal",
            "espn",
            "bbc-news"]
        
        self.widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
        ]
    
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
    
    @timing
    def fetch_data_daily(self, date, language='en', sort_by=None, page_size=100):
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
    
    def change_column_names(self, df):
        return df.rename({'urlToImage' : 'url_to_image', 'publishedAt' : 'published_at'}, axis=1)
    
    @timing
    def append_top_articles(self, df_all):
        df_top = self.fetch_top_articles()
        top_list = []
        for url in df_all['url']:
            if url in df_top['url'].unique():
                top_list.append(1)
            else:
                top_list.append(0)
        df_all["top_article"] = top_list
        return (df_all, len(top_list))
    
    @timing
    def append_facebook_engagement(self, df):
        fe_list = []
        api_key_counter=1
        for url in progressbar.progressbar(df['url'], redirect_stdout=True, widgets=self.widgets):
            fe = self.sharedcount.get_facebook_engagement(url)
            if 'error' in fe.keys():
                self.sharedcount.change_token(self.fb_api_keys[api_key_counter])
                api_key_counter += 1
                print('Error - API limit reached \n    * API changed & Row processed once again')
                fe = self.sharedcount.get_facebook_engagement(url)
            # https://stackoverflow.com/questions/14092989/facebook-api-4-application-request-limit-reached
            time.sleep(1)
            fe_list.append(flatten(fe))
        df_fe = pd.DataFrame(fe_list)
        print("\n\n --------------")
        return df.join(df_fe.set_index('id'), on='url')
    
    @timing
    def drop_duplicates(self, df):
        df = df.groupby('url', group_keys=False).apply(lambda x: x.loc[x.top_article.idxmax()])
        return df
    
    def save_metadata(self, metadata_tuple, date):
        metadata_dict = {
            'fetch_time': metadata_tuple[0],
            'append_top_time': metadata_tuple[1],
            'top_count': metadata_tuple[2],
            'append_social_time': metadata_tuple[3],
            'drop_duplicates_time': metadata_tuple[4],
        }
        with open('Data/metadata_{date:%Y-%m-%d_%H:%M:%S}.json'.format(date=date), 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
    
    @timing
    def harvest_daily(self):
        # print("Fetching data ... ")
        # df, fetch_time = self.fetch_data_daily(datetime.date.today())
        # print("Amount of fetched data: {}".format(df.count())+'\n\n --------------')
        # df.to_csv("Data/data_all.csv")
        
        fetch_time = 0
        
        df = pd.read_csv("Data/data_all.csv")
        print("Data enrichment with top articles ... \n\n --------------")
        df_with_top_count, append_top_time = self.append_top_articles(df)
        df = df_with_top_count[0]
        top_count = df_with_top_count[1]
        print("Data enrichment with facebook social shares ... (this process may take some time)")
        df, append_social_time = self.append_facebook_engagement(df)
        print("Drop duplicates with max value of top_article column ... \n\n --------------")
        df, drop_duplicates_time = self.drop_duplicates(df)
        print("Saving dataframe to Data/ directory ... \n\n --------------")
        date = datetime.datetime.now()
        df.to_csv("Data/data_all_{date:%Y-%m-%d_%H:%M:%S}.csv".format(date=date))
        self.save_metadata((fetch_time, append_top_time, top_count, append_social_time, drop_duplicates_time), date)