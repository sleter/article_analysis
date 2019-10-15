from abc import ABC, abstractmethod

class AbstractNN(ABC):
    @abstractmethod
    def __init__(self, submodule_name):
        # print(__class__)
        print("Initializing ML submodule: {} \n -----------------------------".format(submodule_name))
        
    @abstractmethod
    def read_dataset(self, filename):
        print("Reading {} dataset \n -----------------------------".format(filename))
        
    def split_dataset(self, df, Y_name="top_article"):    
        dataset_X = df.loc[:, df.columns != Y_name].values
        dataset_Y = df[Y_name].values
        X = dataset_X.astype(float)
        Y = dataset_Y
        return X, Y, dataset_X.shape[1]
