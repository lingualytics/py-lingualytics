
# Lingualytics : Easy codemixed analytics
![](https://img.shields.io/github/issues-raw/lingualytics/py-lingualytics?style=flat-square)
![](https://img.shields.io/pypi/dm/lingualytics?style=flat-square)
![](https://img.shields.io/website?url=https%3A%2F%2Flingualytics.tech%2F&style=flat-square)
![](https://img.shields.io/pypi/v/lingualytics?style=flat-square)
![](https://img.shields.io/pypi/status/lingualytics?style=flat-square&label=stage)
![](https://img.shields.io/github/languages/count/lingualytics/py-lingualytics?style=flat-square)
![](https://img.shields.io/github/languages/code-size/lingualytics/py-lingualytics?style=flat-square)
![](https://img.shields.io/librariesio/github/lingualytics/py-lingualytics?style=flat-square)
![](https://img.shields.io/github/license/lingualytics/py-lingualytics?style=flat-square)

Lingualytics is a Python library for dealing with code mixed text.  
Lingualytics is powered by powerful libraries like [Pytorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers), [Texthero](https://texthero.org/), [NLTK](http://www.nltk.org/) and [Scikit-learn](https://scikit-learn.org/).

![train-demo](github/train-demo.gif)

## 🌟 Features

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

## 🧠 Pretrained Models

Checkout some codemix friendly models that we have trained using Lingualytics

- [bert-base-multilingual-codemixed-cased-sentiment](https://huggingface.co/rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment)
- [bert-base-en-es-codemix-cased](https://huggingface.co/rohanrajpal/bert-base-en-es-codemix-cased)
- [bert-base-en-hi-codemix-cased](https://huggingface.co/rohanrajpal/bert-base-en-hi-codemix-cased)

## 💾 Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install lingualytics.

```bash
pip install lingualytics
```

## 🕹️ Usage

### Preprocessing
}}}}{
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

learner = Learner(model_type = 'bert',
                model_name = 'bert-base-multilingual-cased',
                dataset = 'SAIL-2017')
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

## 👪 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## ⚖️ License

[MIT](https://choosealicense.com/licenses/mit/)

## 📚 References

1. Khanuja, Simran, et al. "GLUECoS: An Evaluation Benchmark for Code-Switched NLP." arXiv preprint arXiv:2004.12376 (2020).
