# Lingualytics : Easy codemixed analytics

Lingualytics is a Python library for dealing with code mixed text.  
Lingualytics is powered by powerful libraries like [Pytorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers), [Texthero](https://texthero.org/), [NLTK](http://www.nltk.org/) and [Scikit-learn](https://scikit-learn.org/).

## Features

1. Preprocessing
    - Remove stopwords
    - Remove punctuations, with an option to add punctuations of your own language
    - Remove words less than a character limit

2. Representation
    - Find n-grams from given text

3. NLP
    - Classification using PyTorch
        - Train a classifier on your data to perform tasks like Sentiment Analysis
        - Evaluate the classifier with metrics like accuracy, f1 score, precision and recall
        - Use the trained tokenizer to tokenize text
    - Some pretrained Huggingface models trained on codemixed datasets you can use
        - [bert-base-multilingual-codemixed-cased-sentiment](https://huggingface.co/rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install lingualytics.

```bash
pip install lingualytics
```

## Usage

### Preprocessing

```python
from lingualytics.preprocessing import remove_lessthan, remove_punctuation, remove_stopwords
from lingualytics.stopwords import hi_stopwords,en_stopwords
from texthero.preprocessing import remove_digits
import pandas as pd
df = pd.read_csv(
   "https://github.com/lingualytics/py-lingualytics/raw/master/datasets/SAIL_2017/Processed_Data/Devanagari/validation.txt", header=None, sep='\t', names=['text','label']
)
# pd.set_option('display.max_colwidth', None)
df['clean_text'] = df['text'].pipe(remove_digits) \
                                    .pipe(remove_punctuation) \
                                    .pipe(remove_lessthan,length=3) \
                                    .pipe(remove_stopwords,stopwords=en_stopwords.union(hi_stopwords))
print(df)
```

### Classification

The train data path should have 4 files
    - train.txt
    - validation.txt
    - test.txt

You can just download `datasets/SAIL_2017/Processed Data/Devanagari` from the Github repository to try this out.

```python
from lingualytics.learner import Learner

learner = Learner(data_dir='<path-to-train-data>',
                output_dir='<path-to-output-predictions-and-save-the-model>')
learner.fit()
```

### Find topmost n-grams

```python
from lingualytics.representation import get_ngrams
import pandas as pd
df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

ngrams = get_ngrams(df['text'],n=2)

print(ngrams[:10])
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
