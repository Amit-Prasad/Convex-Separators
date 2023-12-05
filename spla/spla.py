import numpy as np
from k_fold_cross_validation import *
from preprocessing import get_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from spla1 import test, get_probs, train_spla_2
import sys

np.random.seed(1)

def count_pararmeters(W):
    return W.shape[0] * W.shape[1], W.shape[1]

def spla_run(data_name):
    output_dict = {}
    output_dict['dataset'] = data_name
    output_dict['algo'] = 'spla'
    data_x, data_y, data_x_test, data_y_test = get_dataset(data_name)
    output_dict['train_examples'] = data_x.shape[0]
    output_dict['columns'] = data_x.shape[1]
    output_dict['test_examples'] = data_x_test.shape[0]
    output_dict['positive_train_examples'] = np.sum(data_y)
    output_dict['positive_test_examples'] = np.sum(data_y_test)
    output_dict['negative_train_examples'] = len(data_y) - np.sum(data_y)
    output_dict['negative_test_examples'] = len(data_y_test) - np.sum(data_y_test)
    data_x_temp = np.zeros((data_x.shape[0], data_x.shape[1]+1))
    data_x_temp[:, 0:-1] = data_x
    data_x_temp[:, -1] = np.ones(data_x.shape[0])
    data_x = data_x_temp
    data_x_temp = np.zeros((data_x_test.shape[0], data_x_test.shape[1]+1))
    data_x_temp[:, 0:-1] = data_x_test
    data_x_temp[:, -1] = np.ones(data_x_test.shape[0])
    data_x_test = data_x_temp
    clf, mean, std_dev, ter_codes = k_fold_cross_validate(data_x, data_y, 60, k=5, scoring='balanced_accuracy')
    output_dict['k_fold_ter_codes'] = ter_codes
    output_dict['mean'] = mean
    output_dict['std_dev'] = std_dev
    clf, ter_code = train(data_x.T, data_y, 60, visualise = False)
    output_dict['main_train_ter_code'] = ter_code
    c_n, c_h = count_pararmeters(clf)
    output_dict['number_of_hyperplanes'] = c_h
    output_dict['number_of_parameters'] = c_n
    predicted_labels = test(clf, data_x.T)
    predicted_labels_test = test(clf, data_x_test.T)
    output_dict['train_accuracy'] = accuracy_score(data_y, predicted_labels)
    output_dict['train_balanced_accuracy'] = balanced_accuracy_score(data_y, predicted_labels)
    output_dict['train_recall'] = recall_score(data_y, predicted_labels)
    output_dict['train_precision'] = precision_score(data_y, predicted_labels)
    output_dict['train_f1_score'] = f1_score(data_y, predicted_labels)
    cf_train = confusion_matrix(data_y, predicted_labels)
    output_dict['train_true_positive'] = cf_train[1, 1]
    output_dict['train_false_positive'] = cf_train[0, 1]
    output_dict['train_false_negative'] = cf_train[1, 0]
    output_dict['train_true_negative'] = cf_train[0, 0]
    output_dict['test_accuracy'] = accuracy_score(data_y_test, predicted_labels_test)
    output_dict['test_balanced_accuracy'] = balanced_accuracy_score(data_y_test, predicted_labels_test)
    output_dict['test_recall'] = recall_score(data_y_test, predicted_labels_test)
    output_dict['test_precision'] = precision_score(data_y_test, predicted_labels_test)
    output_dict['test_f1_score'] = f1_score(data_y_test, predicted_labels_test)
    cf_test = confusion_matrix(data_y_test, predicted_labels_test)
    output_dict['test_true_positive'] = cf_test[1, 1]
    output_dict['test_false_positive'] = cf_test[0, 1]
    output_dict['test_false_negative'] = cf_test[1, 0]
    output_dict['test_true_negative'] = cf_test[0, 0]
    roc_train = roc_auc_score(data_y, get_probs(clf, data_x.T))
    roc_test = roc_auc_score(data_y_test, get_probs(clf, data_x_test.T))
    output_dict['roc_train'] = roc_train
    output_dict['roc_test'] = roc_test
    print(output_dict)


spla_run(sys.argv[1])