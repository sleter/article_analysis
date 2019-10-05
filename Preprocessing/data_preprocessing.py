import pandas as pd
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from utils.helpers import timing
from sklearn.manifold import TSNE
from wordcloud import WordCloud

class DataPreprocessing():
    def __init__(self, filename):
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
    
    def save_embeddings(self):
        (df, found_word_count, not_found_word_count, time_tokens), time_embeddings = self.create_embeddings()
        df.to_csv('GoogleNewsModelData/EmbeddingsData/embeddings_{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.datetime.now()))
        metadata_dict = {
            'time_of_creating_tokens': time_tokens,
            'time_of_creating_embeddings': time_embeddings,
            'found_word_count': found_word_count,
            'not_found_word_count': not_found_word_count,
        }
        with open('GoogleNewsModelData/EmbeddingsData/metadata_{date:%Y-%m-%d_%H:%M:%S}.json'.format(date=datetime.datetime.now()), 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
    
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
        
        