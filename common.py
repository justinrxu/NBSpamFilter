from sklearn.metrics import accuracy_score
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)



def kfsplit(X, k):
    fold_indices = np.array_split(np.arange(X.shape[0], dtype=int), k)
    train_test_indices = []
    for i in range(k):
        train_indices = np.array([])
        test_indices = []
        for fold in range(len(fold_indices)):
            if fold == i:
                test_indices = fold_indices[fold]
            else:
                train_indices = np.concatenate((train_indices.astype(int), fold_indices[fold]))
        train_test_indices.append((train_indices, test_indices))
    return train_test_indices


def my_cross_val(method, X, y, k):
    accuracy = []
    for train_index, test_index in kfsplit(X, k):
        method.fit(X[train_index], y[train_index])
        y_pred = method.predict(X[test_index])
        accuracy.append(accuracy_score(y_pred, y[test_index]))
        print("Fold {}: {}".format(len(accuracy), accuracy[len(accuracy) - 1]))
    print("Mean: {}".format(np.mean(np.array(accuracy))))
    print("Standard Deviation: {}".format(np.std(np.array(accuracy))))
