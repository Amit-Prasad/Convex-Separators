import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from spla1 import train, test, train_1, train_spla_2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import statistics
import time


def k_fold_cross_validate(data_x, data_y, n_hyp, k=5, scoring = 'accuracy'):
    classifiers = []
    scores=[]
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    ter_codes = []
    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(data_x, data_y):
        # select rows
        train_x, val_x = data_x[train_ix], data_x[test_ix]
        train_y, val_y = data_y[train_ix], data_y[test_ix]

        #clf, ter_code = train_1(train_x.T, train_y, n_hyp, visualise=False)
        clf, ter_code = train(train_x.T, train_y, n_hyp, visualise=False)
        ter_codes.append(ter_code)
        predicted_labels = test(clf, val_x.T)
        score = 0.0
        if scoring == 'accuracy':
            score = accuracy_score(val_y, predicted_labels)
        elif scoring == 'precision':
            score = precision_score(val_y, predicted_labels)
        elif scoring == 'recall':
            score = recall_score(val_y, predicted_labels)
        elif scoring == 'balanced_accuracy':
            score = balanced_accuracy_score(val_y, predicted_labels)
        elif scoring == 'f1_score':
            score = f1_score(val_y, predicted_labels)
        classifiers.append(clf)
        scores.append(score)
    best_index = scores.index(max(scores))
    return classifiers[best_index], statistics.mean(scores), statistics.stdev(scores), ter_codes



