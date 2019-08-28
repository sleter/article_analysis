import requests
import json

class SharedCountApiClient():
    """
    SharedCount API client.
    
    Raises:
        NoneApiKeyError: Raised when API key was not provided
    
    """
    def __init__(self, social_share_api_key=None, facebook_graph_api_key=None):
        if (social_share_api_key or facebook_graph_api_key) is None:
            raise NoneApiKeyError
        else:
            self.api_key = social_share_api_key
            self.facebook_graph_api_key = facebook_graph_api_key
    
    def change_token(self, facebook_graph_api_key):
        self.facebook_graph_api_key = facebook_graph_api_key
    
    def get_counts(self, url : str):
        payload = {
            'url': url,
            'apikey': self.api_key 
        }
        r = requests.get("https://api.sharedcount.com/v1.0/", params=payload)
        data = json.loads(r.text)
        return (url, data)
    
    def get_multiple_counts(self, urls : list):
        return [self.get_counts(url) for url in urls]            

    def show_quota(self):
        r = requests.get("https://api.sharedcount.com/v1.0/quota", params={'apikey': self.api_key })
        data = json.loads(r.text)
        print(data)

    def show_usage(self):
        r = requests.get("https://api.sharedcount.com/v1.0/usage", params={'apikey': self.api_key })
        data = json.loads(r.text)
        print(data)

    def show_api_status(self):
        r = requests.get("https://api.sharedcount.com/v1.0/status")
        data = json.loads(r.text)
        print(data)
        
    def get_facebook_engagement(self, url):
        params = (
            ('id', url),
            ('fields', 'engagement'),
            ('access_token', self.facebook_graph_api_key),)
        r = requests.get('https://graph.facebook.com/v4.0/', params=params)
        return json.loads(r.text)
    
class NoneApiKeyError(Exception):
    """
    Raised when API key was not provided. 
    """
    pass
    