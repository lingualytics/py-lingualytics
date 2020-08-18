import pandas as pd
import collections

def calc_ngrams(text: str, n: int):
    return zip(*[text[i:] for i in range(n)])

def get_ngrams(s: pd.Series, n: int, delimiter: str = ' ', merge: bool = False):
    temp = calc_ngrams(s.str.cat(sep=delimiter).split(delimiter), n)
    ngrams = collections.Counter(temp).most_common()
    if merge:
        ngrams = [(delimiter.join(i[0]),i[1]) for i in ngrams]
    return ngrams
