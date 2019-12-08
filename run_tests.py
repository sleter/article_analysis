import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

from utils.helpers import custom_len

from MachineLearningModels.nn import Simple_NN, Complex_NN
from MachineLearningModels.lstm import Tensorflow_LSTM

# from DataHarvesting.data_harvester import DataHarvester
from Preprocessing.data_preprocessing import DataPreprocessing

class TestModule():
    def __init__(self):
        pass

    def test_snn(self):
        nn = Simple_NN(version="v03", filename="data_21154samples_2019-11-10_22:06:07")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = nn.fit_model(save=False ,epochs=20, batch_size=20, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time[1])+' seconds\n\n')
        # Reading and splitting dataset
        # Choosing set of hyperparameters
        # Searching for best results using GridSearch
        # Saving all hyperparameters with their results
        # Using best parameters to fit, evaluate and save model
        time = nn.fit_optimize_eval_model()
        print("fit_optimize_eval_model process took: {}".format(time[1])+' seconds\n\n')

    def test_cnn(self):
        cnn = Complex_NN(version="v03", filename="data_21154samples_2019-11-10_22:06:07")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = cnn.fit_model(save=False ,epochs=20, batch_size=20, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time[1])+' seconds\n\n')
        # Reading and splitting dataset
        # Choosing set of hyperparameters
        # Searching for best results using GridSearch
        # Saving all hyperparameters with their results
        # Using best parameters to fit, evaluate and save model
        time = cnn.fit_optimize_eval_model()
        print("fit_optimize_eval_model process took: {}".format(time[1])+' seconds\n\n')

    def test_lstm(self, save=True):
        lstm = Tensorflow_LSTM(version="v02", filename="data_19016_lstm_samples_2019-11-10_21:49:27")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = lstm.fit_model(save=save, epochs=10, batch_size=20, optimizer = 'adam',init = 'uniform')
        print("fit_model process took: {}".format(time[1])+' seconds\n\n')
        # Optimizing hyperparameters for model and saving these
        # lstm.optimize_model()
        

    # def test_getting_daily_articles(self):
    #     dh = DataHarvester()    
    #     harvest_time = dh.harvest_daily()
    #     print("Process took: {}".format(harvest_time)+' seconds\n\n')

    # def test_gathering_all_available_data(self):
    #     dh = DataHarvester()
    #     gathering_time = dh.gather_all()   
    #     print("Process took: {}".format(gathering_time)+' seconds\n\n')

    # def test_preprocessing(self):
        # dp = DataPreprocessing(filename='data_gathered_2019-09-03-2019-11-04_23535')
    
    def test_create_lstm_samples(self):
        dp = DataPreprocessing(filename='Data/GatheredData/data_gathered_2019-09-03-2019-11-04_23535')
        dp.create_samples(filename="Data/GatheredData/data_gathered_2019-09-03-2019-11-04_23535", embeddings=False)

    def test_create_nn_samples(self):
        dp = DataPreprocessing(filename='GatheredData/data_gathered_2019-09-03-2019-11-04_23535')
        filename = dp.save_embeddings(columns=True, word=False)
        # then run create_samples function with csv created with save_embeddings
        dp.create_samples(filename=filename, embeddings=True)

    def publisher_top_articles_predictions_scenario(self, custom_data={}):
        # Calculate metadate averages
        df_all = pd.read_csv('Data/PreprocessedData/data_19016_lstm_samples_2019-11-10_21:49:27.csv', index_col=0)
        df_metadata = df_all.drop(columns=['title', 'top_article'], axis=1)
        averages = np.array(df_metadata.mean(axis=0).tolist())

        # Using day that wasn't used in training or evaluating model
        df_day = pd.read_csv('Data/data_all_2019-11-04_200347.csv', index_col=0)

        #### READ part of sample lstm to check predictions

        df_day = df_day.sort_values('top_article', ascending=False).drop_duplicates('title').sort_index()

        print(df_day['title'].value_counts())

        df_day = df_day.dropna(subset=['top_article', 'title'])

        df_day = df_day[['title', 'top_article']]
        print(df_day['top_article'].value_counts())

        def transform_column(column_df):
            col_name = column_df.name
            column_list = [str(title) for title in column_df]
            tokens = [word_tokenize(title) for title in column_list]
            tokens = [[word.lower() for word in title if word.isalpha()] for title in tokens]
            stopwords_list = set(stopwords.words('english'))
            tokens = [[word for word in title if not word in stopwords_list] for title in tokens]
            tokens = [' '.join(title) for title in tokens]
            return pd.Series(tokens), custom_len(tokens)

        title_series, maxlen = transform_column(column_df=df_day['title'])
        df_day.reset_index(inplace=True, drop=True)
        df_day['title_preprocessed'] = title_series
        print(df_day['title_preprocessed'].value_counts())

        # Create function that split loads with 10 top articles and 10 not top articles
        # then transform these as in split_dataset fuction in nn_abc
        def create_predict_sample(df):
            X = df['title_preprocessed'].values
            y = df['top_article'].values
            encoded_titles_X = []
            for title in X:
                encoded_titles_X.append(one_hot(title, maxlen))
            padded_titles_X = pad_sequences(encoded_titles_X, maxlen=7, padding='post')
            return padded_titles_X, y

        X, y = create_predict_sample(df_day)

        arrays = [averages for _ in range(X.shape[0])]
        averages = np.stack(arrays, axis=0)

        # Read model
        model = tf.keras.models.load_model('MachineLearningModels/SavedModels/m.tensorflow_lstm_2019-11-17_v02.h5')

        # Predict 
        predictions = model.predict([X, averages])
        # predictions = predictions.argmax(axis=-1)
        df_day['predictions'] = predictions
        df_day.sort_values('predictions', ascending=False, inplace=True)
        print(df_day.head(50))
        # loss, accuracy, auc, precision, recall = model.evaluate([X, averages], y)
        # print("\nINFO")
        # print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))  




def main():
    tm = TestModule()
    # tm.test_snn()
    # tm.test_cnn()
    tm.test_lstm()
    # tm.test_create_samples()
    # tm.test_create_nn_samples()
    # tm.publisher_top_articles_predictions_scenario()
    # tm.test_lstm()
    # tm.test_gathering_all_available_data()

if __name__== "__main__":
  main()

