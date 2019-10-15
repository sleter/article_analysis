import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer():
    def __init__(self, samples_filename="data_9415samples_2019-10-15_15:06:32.csv"):
        self.samples_df = pd.read_csv('Data/PreprocessedData/'+samples_filename, index_col=0)

    
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
            fig.savefig("corr_matrix.png")
    