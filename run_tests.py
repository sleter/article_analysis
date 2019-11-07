from MachineLearningModels.nn import Simple_NN, Complex_NN
from MachineLearningModels.lstm import Tensorflow_LSTM

from DataHarvesting.data_harvester import DataHarvester
from Preprocessing.data_preprocessing import DataPreprocessing

class TestModule():
    def __init__(self):
        pass

    def test_snn(self):
        nn = Simple_NN(version="v03", filename="data_21154samples_2019-11-05_163624")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = nn.fit_model(save=False ,epochs=20, batch_size=20, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time)+' seconds\n\n')
        # Reading and splitting dataset
        # Choosing set of hyperparameters
        # Searching for best results using GridSearch
        # Saving all hyperparameters with their results
        # Using best parameters to fit, evaluate and save model
        time = nn.fit_optimize_eval_model()
        print("fit_optimize_eval_model process took: {}".format(time)+' seconds\n\n')

    def test_cnn(self):
        cnn = Complex_NN(version="v03", filename="data_21154samples_2019-11-05_163624")
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = cnn.fit_model(save=False ,epochs=20, batch_size=20, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time)+' seconds\n\n')
        # Reading and splitting dataset
        # Choosing set of hyperparameters
        # Searching for best results using GridSearch
        # Saving all hyperparameters with their results
        # Using best parameters to fit, evaluate and save model
        time = cnn.fit_optimize_eval_model()
        print("fit_optimize_eval_model process took: {}".format(time)+' seconds\n\n')

    def test_lstm(self):
        lstm = Tensorflow_LSTM(version="v02", filename="data_19016_lstm_samples_2019-11-05_163146", embed_size=300, max_word_len=50)
        # Reading and splitting dataset
        # Creating, fitting and evaluating model (using custom hyperparameters)
        time = lstm.fit_model(save=False, epochs=20, batch_size=20, optimizer = 'adam',init = 'glorot_uniform')
        print("fit_model process took: {}".format(time)+' seconds\n\n')

    def test_getting_daily_articles(self):
        dh = DataHarvester()    
        harvest_time = dh.harvest_daily()
        print("Process took: {}".format(harvest_time)+' seconds\n\n')

    def test_gathering_all_available_data(self):
        dh = DataHarvester()
        gathering_time = dh.gather_all()   
        print("Process took: {}".format(gathering_time)+' seconds\n\n')

    def test_preprocessing(self):
        dp = DataPreprocessing(filename='data_gathered_2019-09-03-2019-11-04_23535')


def main():
    tm = TestModule()
    
if __name__== "__main__":
  main()

