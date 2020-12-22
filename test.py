import NaiveBayes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def read_df(csv, method):
    data = pd.read_csv(csv, header=0)
    return method.fit_transform(data.text).toarray(), data.spam.to_numpy()


methods = {
    'CountVectorizer MultinomialNB with Stemmer': NaiveBayes.VectorizedNB(stemmer='Snowball'),
    'CountVectorizer MultinomialNB with Snowball, Stop_Words': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True),
    'CountVectorizerMultinomialNB with Snowball and Stop_Words, n_grams=2': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, n_grams=2),

    'TfidfVectorizer MultinomialNB with Stemmer': NaiveBayes.VectorizedNB(stemmer='Snowball'),
    'TfidfVectorizer MultinomialNB with Snowball, Stop_Words': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, vectorizer='Tfidf'),
    'TfidfVectorizer MultinomialNB with Snowball, Stop_Words, n_grams=2': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, vectorizer='Tfidf'),

    'CountVectorizer GaussianNB with Stemmer': NaiveBayes.VectorizedNB(stemmer='Snowball', classifier='Gaussian'),
    'CountVectorizer GaussianNB with Snowball, Stop_Words': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, classifier='Gaussian'),
    'CountVectorizer GaussianNB with Snowball and Stop_Words, n_grams=2': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, n_grams=2, classifier='Gaussian'),

    'TfidfVectorizer GaussianNB with Stemmer': NaiveBayes.VectorizedNB(stemmer='Snowball', classifier='Gaussian'),
    'TfidfVectorizer GaussianNB with Snowball, Stop_Words': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, vectorizer='Tfidf', classifier='Gaussian'),
    'TfidfVectorizer GaussianNB with Snowball, Stop_Words, n_grams=2': NaiveBayes.VectorizedNB(stemmer='Snowball', stop_words=True, vectorizer='Tfidf', classifier='Gaussian')
}

datasets = ['emails.csv']


def measure_error():
    combinations = [(method, dataset) for method in methods.keys() for dataset in datasets]

    for meth_name, dataset in combinations:
        print("{} with {}:".format(meth_name, dataset))
        method = methods[meth_name]

        X, y = read_df(dataset, method)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        method.classifier.fit(X_train, y_train)
        y_pred = method.predict(X_test)
        print("Correct Predictions: " + str(np.count_nonzero(y_pred == y_test)))
        print("False Positives: " + str(np.count_nonzero(y_pred > y_test)))
        print("Uncaught Spam: " + str(np.count_nonzero(y_pred < y_test)))
        accuracies = cross_val_score(estimator=method.classifier, X=X_train, y=y_train, cv=5)
        print(accuracies)
        print(accuracies.mean())
        print(accuracies.std())


measure_error()