import string
import pandas as pd
import collections

hi_en_punctuation = string.punctuation+'।'+'॥'

def remove_lessthan(s: pd.Series, length: int) -> pd.Series:
    """
    Removes words less than a specific length.

    Parameters
    ----------
    s : pd.Series
        A pandas series.
    length : int
        The minimum length a word should have.
    """
    return s.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))

def remove_punctuation(s: pd.Series, punctuation: str = string.punctuation) -> pd.Series:
    """
    Removes punctuation from the text.
    
    Parameters
    ----------
    s : pd.Series
        A pandas series.
    punctuation : str
        All the punctuation characters you want to remove.
    """
    return s.str.replace(rf"([{punctuation}])+", " ")

def remove_stopwords(s: pd.Series, stopwords: list):
    """
    Removes stopwords from the text.
    
    Parameters
    ----------
    s : pd.Series
        A pandas series.
    stopwords : list of str
        A list of stopwords you want to remove.
    """
    return s.apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

def remove_links(s: pd.Series) -> pd.Series:
    """
    Removes links from the text.

    Parameters
    ----------
    s : pd.Series
        A pandas series.
    """
    return s.str.replace('http\S+|www.\S+', '', case=False)
