import string
import pandas as pd
import collections

def remove_lessthan(s: pd.Series, length: int) -> pd.Series:
    return s.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))

def remove_punctuation(s: pd.Series, punctuation: str = string.punctuation) -> pd.Series:
    return s.str.replace(rf"([{punctuation}])+", " ")

def remove_stopwords(s: pd.Series, stopwords: list):
    return s.apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
