import json

class ConfigReader():
    """
    Class reading hidden config JSON file with API keys
    
    Returns:
        string -- NewsAPI API key or SharedCount API key
    """
    def __init__(self, config_path):
        with open(config_path) as config_file:
            self.data = json.load(config_file)
            
    def get_news_api_key(self):
        return self.data['NEWSAPI_KEY']
    
    def get_sharedcount_api_key(self):
        return self.data['SHAREDCOUNT_KEY']
    
    def get_facebookgraph_api_key(self):
        fb = self.data['FACEBOOKGRAPHAPI_KEY']
        fb1 = self.data['FACEBOOKGRAPHAPI_KEY1']
        fb2 = self.data['FACEBOOKGRAPHAPI_KEY2']
        fb3 = self.data['FACEBOOKGRAPHAPI_KEY3']
        fb4 = self.data['FACEBOOKGRAPHAPI_KEY4']
        return (fb, fb1, fb2, fb3, fb4)