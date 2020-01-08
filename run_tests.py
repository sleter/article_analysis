import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

from utils.helpers import custom_len

from MachineLearningModels.nn import Simple_NN, Complex_NN, Complex_NN_title
from MachineLearningModels.lstm import Tensorflow_LSTM

from DataHarvesting.data_harvester import DataHarvester
from Preprocessing.data_preprocessing import DataPreprocessing
from DataAnalysis.data_analyzer import DataAnalyzer

class TestModule():
    def __init__(self):
        pass

    def test_snn(self):
        nn = Simple_NN(version="v03", filename="data_19734samples_2019-12-17_10_19_43")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = nn.fit_model(save=False ,epochs=20, batch_size=20, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time[1])+' seconds\n\n')
        # Reading and splitting dataset
        # Choosing set of hyperparameters
        # Searching for best results using GridSearch
        # Saving all hyperparameters with their results
        # Using best parameters to fit, evaluate and save model
        time = nn.fit_optimize_eval_model(save=False)
        print("fit_optimize_eval_model process took: {}".format(time[1])+' seconds\n\n')

    def test_cnn(self):
        cnn = Complex_NN(version="v03", filename="data_19734samples_2019-12-17_10_19_43")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = cnn.fit_model(save=True ,epochs=15, batch_size=15, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time[1])+' seconds\n\n')
        # Reading and splitting dataset
        # Choosing set of hyperparameters
        # Searching for best results using GridSearch
        # Saving all hyperparameters with their results
        # Using best parameters to fit, evaluate and save model
        time = cnn.fit_optimize_eval_model()
        print("fit_optimize_eval_model process took: {}".format(time[1])+' seconds\n\n')

    def test_lstm(self, save=True):
        lstm = Tensorflow_LSTM(version="v03", filename="data_16533_lstm_samples_2019-12-10_10_41_58")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = lstm.fit_model(save=save, epochs=12, batch_size=25, optimizer = 'adam',init = 'uniform')
        print("fit_model process took: {}".format(time[1])+' seconds\n\n')
        # Optimizing hyperparameters for model and saving these
        # lstm.optimize_model()
        
    def test_getting_daily_articles(self):
        dh = DataHarvester()    
        harvest_time = dh.harvest_daily()
        print("Process took: {}".format(harvest_time)+' seconds\n\n')

    def test_gathering_all_available_data(self):
        dh = DataHarvester()
        gathering_time = dh.gather_all()   
        print("Process took: {}".format(gathering_time)+' seconds\n\n')

    def test_preprocessing(self):
        dp = DataPreprocessing(filename='GatheredData/data_gathered_2019-09-03-2019-11-04_23535')
        dp.generate_wordcloud(filename="wordcloud2.png")
    
    def test_create_lstm_samples(self):
        dp = DataPreprocessing(filename='GatheredData/data_gathered_2019-09-03-2019-11-04_23535')
        dp.create_samples(filename="Data/GatheredData/data_gathered_2019-09-03-2019-11-04_23535", embeddings=False)

    def test_create_nn_samples(self):
        dp = DataPreprocessing(filename='GatheredData/data_gathered_2019-09-03-2019-11-04_23535')
        filename = dp.save_embeddings(columns=True, word=False)
        # then run create_samples function with csv created with save_embeddings
        dp.create_samples(filename=filename, embeddings=True)

    def publisher_top_articles_predictions_scenario(self, custom_data={}):
        # Calculate metadate averages
        df_all = pd.read_csv('Data/PreprocessedData/data_16533_lstm_samples_2019-12-10_10_41_58.csv', index_col=0)
        df_metadata = df_all.drop(columns=['title', 'top_article'], axis=1)
        # TRY USING MEDIAN INSTEAD OF MEAN
        averages = np.array(df_metadata.mean(axis=0).tolist())
        medians = np.array(df_metadata.median(axis=0).tolist())

        # Using day that wasn't used in training or evaluating model
        df_day = pd.read_csv('Data/data_all_2019-11-10_203224.csv', index_col=0)

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
        model = tf.keras.models.load_model('MachineLearningModels/SavedModels/m.tensorflow_lstm_2019-12-16_v03.h5')

        # Predict 
        predictions = model.predict([X, averages])
        predictions2 = predictions.argmax(axis=-1)

        df_day['predictions'] = predictions
        df_day['predictions2'] = predictions2

        print(df_day.sort_values(by=['predictions2', 'predictions'], ascending=False).head(50))
        print(df_day['predictions2'].value_counts())
        loss, accuracy, auc, precision, recall = model.evaluate([X, averages], y)
        print("\nINFO")
        print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))  
        
    def publisher_top_articles_predictions_scenario2(self, custom_data={}):
        ### 1 
        # dh = DataHarvester()
        # (dh_filepath, time) = dh.gather_all(predefined_filenames=['data_all_2020-01-06_15:41:12.csv'])
        # print(dh_filepath)
        ### 2
        # dh_filepath = "Data/GatheredData/data_gathered_2020-01-06-2020-01-06_684"
        # dp = DataPreprocessing(filename=dh_filepath[5:])
        # filename = dp.save_embeddings(columns=True, word=False)
        # df_day = dp.create_samples(filename=filename, embeddings=True, return_df=True, drop_title=False)

        df_day = pd.read_csv("Data/PreprocessedData/data_583samples_2020-01-07_16_53_07.csv", index_col=0)
        
        # print(df_day.columns)
        unknowns = ['engagement_reaction_count',
        'engagement_comment_count', 'engagement_share_count',
        'engagement_comment_plugin_count',
        'publish_harvest_time_period',
        'engagement_in_time']
        for u in unknowns:
            df_day[u] = df_day[u].median()
            # df_day[u] = df_day[u].mean()

        X_list = [i for i in df_day.columns if i not in ["title", "top_article"]]
        X, y = df_day[X_list].values, df_day["top_article"].values

        model = tf.keras.models.load_model("MachineLearningModels/SavedModels/simple_nn_2020-01-07_v03.h5")
        # loss, accuracy, auc, precision, recall = model.evaluate(X, y)
        # print("\n\nEvaluation on test set\n")
        # print("loss: {} | accuracy: {} | auc: {} | precision: {} | recall: {}".format(loss, accuracy, auc, precision, recall))
        preds = model.predict(X)

        df_out = df_day[["title", "top_article"]]
        df_out["preds"] = preds

        df_out = df_out.sort_values(by=['preds'], ascending=False).head(10)
        print(df_out[["title", "top_article"]])


def main():
    tm = TestModule()
    # Harvesting data
    # tm.test_getting_daily_articles()
    # Gathering data
    # tm.test_gathering_all_available_data()
    # Data preprocessing - generating wordcloud
    # tm.test_preprocessing()
    # Training, optimizing and evaluating model
    #     init = ['glorot_uniform', 'normal'] 
    #     optimizer = ['adam', 'rmsprop']
    #     batch_sizes = [10, 20, 50]
    #     epochs = [10, 15]
    # tm.test_snn()
    # Publisher scenario
    tm.publisher_top_articles_predictions_scenario2()

    ### -----------------------------------------------
    # tm.test_snn()
    # tm.test_cnn()
    # tm.publisher_top_articles_predictions_scenario2()
    # tm.test_lstm()
    # tm.test_create_lstm_samples()
    # tm.test_create_nn_samples()
    # tm.publisher_top_articles_predictions_scenario()
    # tm.test_lstm()
    # tm.test_gathering_all_available_data()
    # da = DataAnalyzer(filename_path="Data/GatheredData/data_gathered_2019-09-03-2019-11-04_23535")
    # da.articles_per_publisher()

    # da = DataAnalyzer(filename_path="Data/GatheredData/data_gathered_2019-09-03-2019-11-04_23535")
    # print(da.analyze_harvest_metadata())

if __name__== "__main__":
  main()

