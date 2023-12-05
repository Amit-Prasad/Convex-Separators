from k_fold_cross_validation import *
from preprocessing import get_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from convex_separator import test, get_probs
from helpers import feature_engineer_rbf, feature_engineer_rbf_sampled, featurewise_normalisation, append_bias
import sys

np.random.seed(2)

def count_pararmeters(W):
    return W.shape[0] * W.shape[1], W.shape[1]

def cs_run(data_name, feature, inv, norm, boundary, bic_enable, balanced = False, regularised = 'l1', c = 0.001):
    '''feature = 0 implies no feature engineering, feature = 1 implies RBF features wrt closest opposite class pairs, feature 2 implies quadratic'''
    output_dict = {}
    feature = int(feature)
    inv = int(inv)
    norm = int(norm)
    boundary = int(boundary)
    bic_enable = int(bic_enable)
    output_dict['dataset'] = data_name
    if feature == 0 and inv == 0:
        output_dict['algo'] = 'cs'
    elif feature == 0 and inv == 1:
        output_dict['algo'] = 'cs_inv'
    elif feature == 1 and inv == 0 and norm == 0:
        output_dict['algo'] = 'cs_rbf'
    elif feature == 1 and inv == 1 and norm == 0:
        output_dict['algo'] = 'cs_rbf_inv'
    elif feature == 1 and inv == 0 and norm == 1:
        output_dict['algo'] = 'cs_rbf_norm'
    elif feature == 1 and inv == 1 and norm == 1:
        output_dict['algo'] = 'cs_rbf_norm_inv'
    elif feature == 2 and inv == 0 and norm == 0:
        output_dict['algo'] = 'cs_qua'
    elif feature == 2 and inv == 1 and norm == 0:
        output_dict['algo'] = 'cs_qua_inv'
    elif feature == 2 and inv == 0 and norm == 1:
        output_dict['algo'] = 'cs_qua_norm'
    elif feature == 2 and inv == 1 and norm == 1:
        output_dict['algo'] = 'cs_qua_norm_inv'

    data_x, data_y, data_x_test, data_y_test = get_dataset(data_name, feature, norm)
    # make the large numbered class as positive
    output_dict['train_examples'] = data_x.shape[0]
    output_dict['columns'] = data_x.shape[1]
    output_dict['test_examples'] = data_x_test.shape[0]
    output_dict['positive_train_examples'] = np.sum(data_y)
    output_dict['positive_test_examples'] = np.sum(data_y_test)
    output_dict['negative_train_examples'] = len(data_y) - np.sum(data_y)
    output_dict['negative_test_examples'] = len(data_y_test) - np.sum(data_y_test)

    if inv:
        data_y = np.where(data_y == 1, 0, 1)
        data_y_test = np.where(data_y_test == 1, 0, 1)

    # print(np.sum(np.where(data_y == 1, 1, 0)))
    # print(np.sum(np.where(data_y == 0, 1, 0)))
    max_hyperplanes = [80]
    if bic_enable:
        max_hyperplanes = [80]
    lambdas = [0]#, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    data_x_copy = np.copy(data_x)
    data_y_copy = np.copy(data_y)
    clf, best_par, ter_codes, num_hyp = grid_search(max_hyperplanes, lambdas, data_x_copy, data_y_copy, feature, norm, boundary, balanced, regularised, c, bic_enable, scoring = 'balanced_accuracy')
    output_dict['grid_search_ter_codes'] = ter_codes
    output_dict['best_params'] = best_par
    output_dict['grid_search_num_hyp'] = num_hyp
    clf, mean, std_dev, ter_codes, num_hyp = k_fold_cross_validate(data_x_copy, data_y_copy, best_par[0], best_par[1], feature, norm, boundary, balanced, regularised, c, bic_enable, k=5, scoring='balanced_accuracy')
    output_dict['k_fold_ter_codes'] = ter_codes
    output_dict['mean'] = mean
    output_dict['std_dev'] = std_dev
    output_dict['k_fold_num_hyp'] = num_hyp
    data_x_copy = data_x_copy.T
    data_x_test = data_x_test.T

    if feature == 1:
        data_x_copy, feature_dots = feature_engineer_rbf(data_x_copy, data_y, gamma=1)
        data_x_test, feature_dots = feature_engineer_rbf_sampled(data_x_test, feature_dots, gamma=1)
        if norm:
            data_x_copy = featurewise_normalisation(data_x_copy)
            data_x_test = featurewise_normalisation(data_x_test)

    data_x_copy = append_bias(data_x_copy)
    data_x_test = append_bias(data_x_test)

    data_x = np.copy(data_x_copy)
    data_y = np.copy(data_y_copy)
    clf, ter_code, all_j_theta, par_error, all_bic_vals, all_acc = train_cs2(data_x_copy, data_y_copy, best_par[0], '--', boundary, best_par[1], balanced, regularised, c, bic_enable, visualise=0)
    #clf, ter_code, all_j_theta, par_error, all_bic_vals, all_acc, all_val_acc, _ = train_cs1(data_x_copy, data_y_copy, val_x, val_y, 80, '--', boundary, 0, bic_enable, False, error_threshold, visualise=0)
    output_dict['all_j_theta'] = all_j_theta
    output_dict['par_error'] = par_error
    output_dict['all_bic_vals'] = all_bic_vals
    output_dict['all_accuracies'] = all_acc
    #output_dict['all_val_acc'] = all_val_acc
    output_dict['main_train_ter_code'] = ter_code
    c_n, c_h = count_pararmeters(clf)
    output_dict['number_of_hyperplanes'] = c_h
    output_dict['number_of_parameters'] = c_n
    predicted_labels = test(clf, data_x)
    predicted_labels_test = test(clf, data_x_test)
    if inv:
        data_y = np.where(data_y == 1, 0, 1)
        data_y_test = np.where(data_y_test == 1, 0, 1)
        predicted_labels = np.where(predicted_labels == 1, 0, 1)
        predicted_labels_test = np.where(predicted_labels_test == 1, 0, 1)

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
    probs_train = get_probs(clf, data_x)
    probs_test = get_probs(clf, data_x_test)
    if inv:
        probs_train = 1-probs_train
        probs_test = 1-probs_test
    roc_train = roc_auc_score(data_y, probs_train)
    roc_test = roc_auc_score(data_y_test, probs_test)
    output_dict['roc_train'] = roc_train
    output_dict['roc_test'] = roc_test
    print(output_dict)
cs_run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], False, 'l1', c=0.001)
#cs_run("covtype_bin_sub", 1, 0 ,1 ,1, 1, False, 'none', c = 0.001)
