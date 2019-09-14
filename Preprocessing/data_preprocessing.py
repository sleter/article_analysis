import pandas as pd
import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors 

class DataPreprocessing():
    def __init__(self, filename):
        self.df = pd.read_csv('Data/'+filename, index_col=0)
    
    def create_tokens_from_title(self):
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
        tokens = list(set(tokens))
        print("Created {} tokens.".format(len(tokens)))
        return tokens
    
    def create_embeddings(self, tokens):
        gmodel = KeyedVectors.load_word2vec_format('GoogleNewsModel/GoogleNews-vectors-negative300.bin.gz', binary = True)
        words_with_embeddings = {}
        not_found_word_count = 0
        for token in tokens:
            if token in gmodel.vocab:
                words_with_embeddings[token] = gmodel[token]
            else:
                not_found_word_count +=1
        df = pd.DataFrame.from_dict(words_with_embeddings, orient='index')
        print("Not found words in model: {}".format(not_found_word_count))
        self.save_embeddings(df)
    
    def save_embeddings(self, df):
        df.to_csv('GoogleNewsModel/EmbeddingsData/embeddings_{date:%Y-%m-%d_%H:%M:%S}.csv'.format(date=datetime.datetime.now()))