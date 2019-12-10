import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import os, json
import dateutil.parser

class DataAnalyzer():
    def __init__(self, filename_path="PreprocessedData/data_9415samples_2019-10-15_15:06:32.csv"):
        sdf = pd.read_csv(filename_path, index_col=0)
        sdf.drop_duplicates(subset=["title", "source_id"], keep="last", inplace=True)
        ap_df = sdf["source_name"].value_counts()
        ap_df = ap_df[ap_df > len(sdf.index)*0.001]
        self.publishers = ap_df.index.tolist()
        self.samples_df = sdf[sdf["source_name"].isin(self.publishers)]
        self.save_path = "DataAnalysis/"
    
    def top_articles_percentage(self):
        ta_list = self.samples_df["top_article"].value_counts().tolist()
        top_rows = ta_list[1]
        all_rows = ta_list[0]
        print("Percantage of top articles in whole samples dataset: {0:.2f}%".format((top_rows/all_rows)*100))
        
    def print_columns_as_list(self, df):
        columns_list = list(df.columns)
        print("Columns list: {}".format(columns_list))
    
    def show_correlation_matrix(self, df, show=True, save=False):
        corr_mat = df.corr()
        fig, ax = plt.subplots(figsize=(20, 12)) 
        plot = sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax)
        if show:
            plt.show()
        elif save:
            fig = plot.get_figure()
            fig.savefig(self.save_path+"corr_matrix.png")
    
    def engagement_per_publisher(self):
        epp = []
        df = self.samples_df[self.samples_df.source_name != 'ESPN']
        publishers = list(df['source_name'].unique())
        engagement_columns = ['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']
        for ec in engagement_columns:
            df_eng = df.groupby(['source_name'])[ec].agg('sum')
            epp.append((ec, df_eng.tolist()))
        return publishers, epp

    def articles_per_publisher(self):
        df = self.samples_df
        df = df.dropna(subset=['source_name', 'published_at'])
        def parse_date(date_string):
            do = dateutil.parser.parse(date_string)
            return do.date().isoformat()
        df["published_at_day"] = df['published_at'].apply(lambda x : parse_date(x))
        df_date_pub = df.sort_values(['source_name', 'published_at_day'])
        dates_list = df_date_pub['published_at_day'].unique().tolist()

        df_date_pub = df.groupby(['source_name', 'published_at_day'])['published_at_day'].agg('count')
        list_date_pub = df_date_pub.tolist()
        p_len = len(dates_list)
        ldp_chunks = [list_date_pub[x:x+p_len] for x in range(0, len(list_date_pub), p_len)]
        return self.publishers, ldp_chunks, dates_list

    def top_n_liked_articles(self, n, main_field='engagement_reaction_count', grouping_fields=['title', 'source_name']):
        grouping_fields.append(main_field)
        df_liked = self.samples_df.sort_values([main_field], ascending=False)
        df_liked = df_liked.dropna(subset=grouping_fields+[main_field])
        titles_df = df_liked[grouping_fields].head(n)
        titles_df.rename(columns={main_field: main_field+'_sum'}, inplace=True)
        return titles_df

    def analyze_harvest_metadata(self):
        filenames = [filename for filename in os.listdir('Data/') if filename.startswith("metadata_")]
        fetch_time = []
        append_top_time = []
        top_count = []
        append_social_time = []
        drop_duplicates_time = []
        for file in filenames:
            with open('data.json') as f:
                data = json.load(f)
                fetch_time.append(data['fetch_time'])
                append_top_time.append(data['append_top_time'])
                top_count.append(data['top_count'])
                append_social_time.append(data['append_social_time'])
                drop_duplicates_time.append(data['drop_duplicates_time'])
        return [(min(fetch_time), max(fetch_time), np.mean(fetch_time)), \
                (min(append_top_time), max(append_top_time), np.mean(append_top_time)), \
                (min(top_count), max(top_count), np.mean(top_count)), \
                (min(append_social_time), max(append_social_time), np.mean(append_social_time)), \
                (min(drop_duplicates_time), max(drop_duplicates_time), np.mean(drop_duplicates_time))]
        




    