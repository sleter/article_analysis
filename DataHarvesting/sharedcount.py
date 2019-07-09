import requests
import json

class SharedCountApiClient():
    """
    SharedCount API client.
    
    Raises:
        NoneApiKeyError: Raised when API key was not provided
    
    """
    def __init__(self, api_key=None):
        if api_key is None:
            raise NoneApiKeyError
        else:
            self.api_key = api_key
    
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
    
class NoneApiKeyError(Exception):
    """
    Raised when API key was not provided. 
    """
    pass
    