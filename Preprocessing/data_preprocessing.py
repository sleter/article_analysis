import pandas as pd
import datetime
import dateutil
import json
import pytz
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors, Doc2Vec, doc2vec
from utils.helpers import timing, custom_len
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class DataPreprocessing():
    def __init__(self, filename="GatheredData/data_gathered_2019-09-03-2019-10-03_10437"):
        self.df = pd.read_csv('Data/'+filename, index_col=0)
    
    @timing
    def create_tokens_from_title(self, duplicates=False):
        # On first use ->
        # import nltk
        # nltk.download('punkt')
        # nltk.download('stopwords')
        titles_df = self.df['title']
        titles_list = [str(title) for title in titles_df]
        # Create one long string for processing
        lstring = ' '.join(titles_list)
        # Tokenize string into words
        tokens = word_tokenize(lstring)
        # Remove non-alphabetic tokens and make all characters lowercase. 
        tokens = [word.lower() for word in tokens if word.isalpha()]
        # Remove stop words
        stopwords_list = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stopwords_list]
        # Remove duplicates
        if not duplicates:
            tokens = list(set(tokens))
        print("Created {} tokens.".format(len(tokens)))
        return tokens
    
    @timing
    def create_embeddings(self):
        tokens, time_tokens = self.create_tokens_from_title()
        gmodel = KeyedVectors.load_word2vec_format('GoogleNewsModelData/GoogleNews-vectors-negative300.bin.gz', binary = True)
        words_with_embeddings = {}
        not_found_word_count = 0
        found_word_count = 0
        for token in tokens:
            if token in gmodel.vocab:
                words_with_embeddings[token] = gmodel[token]
                found_word_count +=1
            else:
                not_found_word_count +=1
        df = pd.DataFrame.from_dict(words_with_embeddings, orient='index')
        # print("Number of words found in model: {}".format(found_word_count))
        # print("Number of words not found in model: {}".format(not_found_word_count))
        print("Found embeddings for {:.2%} of words".format(found_word_count/len(tokens)))
        return (df, found_word_count, not_found_word_count, time_tokens)
    

    def transform_column(self, df, column_df, vector_size=50, embedding=True):
        col_name = column_df.name
        column_list = [str(title) for title in column_df]
        tokens = [word_tokenize(title) for title in column_list]
        tokens = [[word.lower() for word in title if word.isalpha()] for title in tokens]
        stopwords_list = set(stopwords.words('english'))
        tokens = [[word for word in title if not word in stopwords_list] for title in tokens]
        if embedding:
            def create_tagged_document(tokens):
                for i, title in enumerate(tokens):
                    yield doc2vec.TaggedDocument(title, [i])
            titles_train_data = list(create_tagged_document(tokens))
            
            model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)
            model.build_vocab(titles_train_data)
            model.train(titles_train_data, total_examples=model.corpus_count, epochs=model.epochs)
            
            titles_vectors = [model.infer_vector(title) for title in tokens]
            for i in range(1,vector_size+1):
                hlist = [vector[i-1] for vector in titles_vectors]
                df['{}_embedding_{}'.format(col_name, i)] = np.asarray(hlist, dtype=np.float32)
            return df, tokens, titles_vectors
        else:
            tokens = [' '.join(title) for title in tokens]
            df['title'] = pd.Series(tokens)
            return df, custom_len(tokens)


    @timing
    def create_columns_embeddings(self):
        df = self.df.copy(deep=True)

        titles_df = df['title']
        dft, tokenst, titles_vectors = self.transform_column(df, column_df=titles_df, vector_size=10)
        content_df = dft['content']
        dfc, tokensc, content_vectors = self.transform_column(dft, column_df=content_df)
        
        dfc = self.add_time_difference_column(dfc)
        
        print("Created {} title tokens.".format(len(titles_vectors)))
        print("Created {} content tokens.".format(len(content_vectors)))
        return (dfc, len(tokenst), len(titles_vectors), len(tokensc), len(content_vectors))

    def add_time_difference_column(self, df):
        def merge(columns_data):
            try:
                published_at = dateutil.parser.parse(columns_data[0])
                harvested_at = datetime.datetime.strptime(columns_data[1], "%Y-%m-%d_%H:%M:%S")
                warsaw = pytz.timezone('Europe/Warsaw')
                harvested_at = harvested_at.astimezone(warsaw)
                time_diff = harvested_at - published_at
            except TypeError:
                return np.nan
            return time_diff.total_seconds() / 60
        df['publish_harvest_time_period'] = df[['published_at', 'harvested_at_date']].apply(lambda x: merge(x), axis=1)
        return df

    def save_embeddings(self, columns=True, word=False):
        if columns:
            (df, title_tokens, titles_vectors, content_tokens, content_vectors), title_time_embeddings = self.create_columns_embeddings()
            metadata_dict = {
                'time_of_creating_title_embeddings': title_time_embeddings,
                'number_of_title_tokens': title_tokens,
                'number_of_titles_vectors': titles_vectors,
                'number_of_content_tokens': content_tokens,
                'number_of_content_vectors': content_vectors,
            }
            filename = 'categorical_embeddings_{date:%Y-%m-%d_%H_%M_%S}.csv'.format(date=datetime.datetime.now())
            df.to_csv('GoogleNewsModelData/EmbeddingsData/'+filename)
        elif word:
            (df, found_word_count, not_found_word_count, time_tokens), time_embeddings = self.create_embeddings()
            metadata_dict = {
                'time_of_creating_tokens': time_tokens,
                'time_of_creating_embeddings': time_embeddings,
                'found_word_count': found_word_count,
                'not_found_word_count': not_found_word_count,
            }
            filename = 'embeddings_{date:%Y-%m-%d_%H_%M_%S}.csv'.format(date=datetime.datetime.now())
            df.to_csv('GoogleNewsModelData/EmbeddingsData/'+filename)
        with open('GoogleNewsModelData/EmbeddingsData/metadata_{date:%Y-%m-%d_%H_%M_%S}.json'.format(date=datetime.datetime.now()), 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
        return filename
    
    @timing     
    def tsne_dim_red(self):
        df_embd = pd.read_csv('GoogleNewsModelData/EmbeddingsData/embeddings_2019-10-05_12:17:49.csv', index_col=0)
        tsne = TSNE(n_components=3 ,perplexity=100, learning_rate=200)
        np_tsne = tsne.fit_transform(df_embd[:100])
        word_list = df_embd.index.values.tolist()
        word_list = word_list[:100]
        tsne_list_1, tsne_list_2, tsne_list_3 = [], [], []
        for elem in np_tsne:
            tsne_list_1.append(elem[0])
            tsne_list_2.append(elem[1])
            tsne_list_3.append(elem[2])
        df = pd.DataFrame({"tsne1": tsne_list_1, "tsne2": tsne_list_2, "tsne3": tsne_list_3}, index=word_list)
        return df
        
    def generate_wordcloud(self, filename="wordcloud.png", number_of_words=200, figsize=(30.0,17.0)):
        tokens, _ = self.create_tokens_from_title(duplicates=True)
        text = ' '.join(map(str, tokens))
        wc = WordCloud(background_color='black', 
                        max_words=number_of_words, 
                        max_font_size=100,
                        random_state=69,
                        width=1280,
                        height=720
                       )
        wc.generate(text)
        plt.figure(figsize=figsize)
        plt.imshow(wc)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("Preprocessing/"+filename)
        
    def create_samples(self, filename, embeddings = True, return_df=False, drop_title=True):
        le = LabelEncoder()
        ohe = OneHotEncoder()
        sc = StandardScaler()

        scale_columns = ['source_id_abc-news', 'source_id_al-jazeera-english',
                'source_id_bbc-news', 
                'source_id_business-insider', 'source_id_cbs-news', 'source_id_cnn',
                'source_id_espn', 'source_id_newsweek', 'source_id_reuters',
                'source_id_the-irish-times',
                'source_id_the-new-york-times',
                'source_id_the-wall-street-journal', 'author']

        if embeddings:
            df = pd.read_csv('GoogleNewsModelData/EmbeddingsData/{}'.format(filename), index_col=0)
            # Drop unwanted duplicates
            df.drop_duplicates(subset=["title", "source_id"], keep="last", inplace=True)
            ap_df = df["source_id"].value_counts()
            ap_df = ap_df[ap_df > len(df.index)*0.001].index.tolist()
            # print(ap_df)
            # Drop unwanted publishers
            df = df[df["source_id"].isin(ap_df)]
            # print(df["source_id"].value_counts())
            # Create column with minutes between publish and harvest time
            df = self.add_time_difference_column(df)
            # Drop not used columns
            columns_to_drop = ["source_name", "description", "title", "url", "url_to_image", "published_at", "content", "harvested_at_date"]
            if not drop_title:
                columns_to_drop.remove("title")
            df = df.drop(columns_to_drop, axis=1)
            # Drop rows without labels
            df = df.dropna(subset=["source_id", 'top_article', 'author'])
            # Fill nan values with mean
            df = df.fillna(df.mean())
            # Change label column (top_article) and other float columns to ints
            df.top_article = df.top_article.astype(int)
            df.engagement_reaction_count = df.engagement_reaction_count.astype(int)
            df.engagement_comment_count = df.engagement_comment_count.astype(int)
            df.engagement_share_count = df.engagement_share_count.astype(int)
            df.engagement_comment_plugin_count = df.engagement_comment_plugin_count.astype(int)
            # Label categorical data
            df = pd.get_dummies(df, columns=['source_id'])
            df['author'] = le.fit_transform(df['author'].astype(str))
            # print("Number of tokens: {}".format(tokens))
            # Add columns
            df['engagement_in_time'] = df['engagement_reaction_count']/df['publish_harvest_time_period']
            # Scale columns
            df[scale_columns] = sc.fit_transform(df[scale_columns])
            df[['engagement_in_time','engagement_reaction_count','engagement_comment_count','engagement_share_count','engagement_comment_plugin_count','publish_harvest_time_period']] = sc.fit_transform(df[['engagement_in_time','engagement_reaction_count','engagement_comment_count','engagement_share_count','engagement_comment_plugin_count','publish_harvest_time_period']])
            # print(df.info())
            filename = 'Data/PreprocessedData/data_{}samples_{date:%Y-%m-%d_%H_%M_%S}.csv'.format(len(df.index),date=datetime.datetime.now())
            df.to_csv(filename)
        else:
            df = pd.read_csv("{}".format(filename), index_col=0)
            # Drop unwanted duplicates
            df.drop_duplicates(subset=["title", "source_id"], keep="last", inplace=True)
            ap_df = df["source_id"].value_counts()
            ap_df = ap_df[ap_df > len(df.index)*0.001].index.tolist()
            # print(ap_df)
            # Drop unwanted publishers
            df = df[df["source_id"].isin(ap_df)]
            # print(df["source_id"].value_counts())
            # Create column with minutes between publish and harvest time
            df = self.add_time_difference_column(df)
            # Drop not used columns
            columns_to_drop = ["source_name", "description", "url", "url_to_image", "published_at", "content", "harvested_at_date"]
            df = df.drop(columns_to_drop, axis=1)
            # Drop rows without labels
            df = df.dropna(subset=["source_id", 'top_article', 'author'])
            # Fill nan values with mean
            df = df.fillna(df.mean())
            # Change label column (top_article) and other float columns to ints
            df.top_article = df.top_article.astype(int)
            df.engagement_reaction_count = df.engagement_reaction_count.astype(int)
            df.engagement_comment_count = df.engagement_comment_count.astype(int)
            df.engagement_share_count = df.engagement_share_count.astype(int)
            df.engagement_comment_plugin_count = df.engagement_comment_plugin_count.astype(int)
            # Label categorical data
            df = pd.get_dummies(df, columns=['source_id'])
            df['author'] = le.fit_transform(df['author'].astype(str))
            # Prepare title column
            df, tokens = self.transform_column(df, column_df=df['title'], embedding=False)
            # Drop titles again if any NaNs were created
            df = df.dropna(subset=['title'])
            # print("Number of tokens: {}".format(tokens))
            # Add columns
            df['engagement_in_time'] = df['engagement_reaction_count']/df['publish_harvest_time_period']
            # Scale columns
            df[scale_columns] = sc.fit_transform(df[scale_columns])
            df[['engagement_in_time','engagement_reaction_count','engagement_comment_count','engagement_share_count','engagement_comment_plugin_count','publish_harvest_time_period']] = sc.fit_transform(df[['engagement_in_time','engagement_reaction_count','engagement_comment_count','engagement_share_count','engagement_comment_plugin_count','publish_harvest_time_period']])
            df.to_csv('Data/PreprocessedData/data_{}_lstm_samples_{date:%Y-%m-%d_%H_%M_%S}.csv'.format(len(df.index),date=datetime.datetime.now()))
        if return_df:
            return df
        