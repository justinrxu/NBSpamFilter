from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import functools

stemmers = {
    'Porter': PorterStemmer(),
    'Snowball': SnowballStemmer('english')
}

vectorizers = {
    'Count': CountVectorizer,
    'Tfidf': TfidfVectorizer
}

classifiers = {
    'Multinomial': MultinomialNB(),
    'Gaussian': GaussianNB()
}

def stemmed_words(words, stemmer, vectorizer):
    try:
        stemmer = stemmers[stemmer]
    except KeyError:
        return 'word'
    analyzer = vectorizer.build_analyzer()
    return [stemmer.stem(word) for word in analyzer(words)]


class VectorizedNB:
    def __init__(self, stemmer='', stop_words=False, vectorizer='Count', n_grams=1, classifier='Gaussian'):
        self.stop_words = stopwords.words('english') if stop_words else None
        self.analyzer = functools.partial(stemmed_words, stemmer=stemmer, vectorizer=vectorizers[vectorizer](ngram_range=(1, n_grams)))
        self.vectorizer = vectorizers[vectorizer](stop_words=self.stop_words, analyzer=self.analyzer, max_features=3000)
        self.classifier = classifiers[classifier] if classifier in classifiers else classifier['Gaussian']

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)

