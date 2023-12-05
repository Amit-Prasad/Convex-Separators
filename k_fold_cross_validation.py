import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from convex_separator import test, train_cs2, train_cs1, train_cs2_k_hyperplane_search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import statistics
from helpers import feature_engineer_rbf, feature_engineer_rbf_sampled, feature_engineer_quadratic, featurewise_normalisation, append_bias

def k_fold_cross_validate(data_x, data_y, n_hyp, lamb, feature, norm, boundary, balanced, regularised, c, bic_enable, k=5, scoring = 'accuracy'):
    classifiers = []
    scores=[]
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    ter_codes = []
    num_hyp = []
    #miss_avg = []
    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(data_x, data_y):
        # select rows
        train_x, val_x = data_x[train_ix], data_x[test_ix]
        train_y, val_y = data_y[train_ix], data_y[test_ix]
        train_x = train_x.T
        val_x = val_x.T
        if feature == 1:
            train_x, feature_dots = feature_engineer_rbf(train_x, train_y, gamma=1)
            val_x, feature_dots = feature_engineer_rbf_sampled(val_x, feature_dots, gamma=1)
            if norm:
                train_x = featurewise_normalisation(train_x)
                val_x = featurewise_normalisation(val_x)

        train_x = append_bias(train_x)
        val_x = append_bias(val_x)
        clf, ter_code, _, _, _, _ = train_cs2(train_x, train_y, n_hyp, '--', boundary, lamb, balanced, regularised, c, bic_enable, visualise=0)
        #clf, ter_code, all_error, all_error_val, all_par_error, all_vals, all_acc, all_acc_vals, miss_drop = train_cs1(train_x, train_y, val_x, val_y, n_hyp, '--', boundary, lamb, bic_enable, 1, 0, visualise=0)
        #miss_avg.append(miss_drop)
        num_hyp.append(clf.shape[1])
        ter_codes.append(ter_code)
        predicted_labels = test(clf, val_x)
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
    return classifiers[best_index], statistics.mean(scores), statistics.stdev(scores), ter_codes, num_hyp#, statistics.mean(miss_avg)

def grid_search(max_hyp, lambdas, data_x, data_y, feature, norm, boundary, balanced, regularised, c, bic_enable, scoring = 'accuracy'):
    params = []
    scores = []
    classifiers = []
    num_hyp = []
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.1, random_state=1)
    train_x = train_x.T
    val_x = val_x.T
    if feature == 1:
        train_x, feature_dots = feature_engineer_rbf(train_x, train_y, gamma=1)
        val_x, feature_dots = feature_engineer_rbf_sampled(val_x, feature_dots, gamma=1)
        if norm:
            train_x = featurewise_normalisation(train_x)
            val_x = featurewise_normalisation(val_x)

    train_x = append_bias(train_x)
    val_x = append_bias(val_x)
    ter_codes = []
    for i in range(len(max_hyp)):
        for j in range(len(lambdas)):
            lamb = lambdas[j]
            n = max_hyp[i]
            train_x_copy = np.copy(train_x)
            train_y_copy = np.copy(train_y)
            clf, ter_code, _, _, _, _ = train_cs2(train_x_copy, train_y_copy, n, '--', boundary, lamb, balanced, regularised, c, bic_enable, visualise=0)
            #clf, ter_code, _, _, _, _ = train_cs1(train_x, train_y, n, '--', boundary, lamb, bic_enable, visualise=0)
            num_hyp.append(clf.shape[1])
            ter_codes.append(ter_code)
            predicted_labels = test(clf, val_x)
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
            params.append((n, lamb))
            scores.append(score)
    best_index = scores.index(max(scores))
    return classifiers[best_index], params[best_index], ter_codes, num_hyp
