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