import collections
from datetime import date, timedelta
from functools import wraps
from time import time

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_dates_between(from_date, to_date):
    delta = to_date - from_date
    days = [from_date+timedelta(days=i) for i in range(delta.days +1)]
    return days
    
def timing(f):
    @wraps(f)
    def wrapper(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, te-ts
    return wrapper