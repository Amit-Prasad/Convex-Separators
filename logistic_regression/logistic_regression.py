import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from preprocessing import get_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy import optimize
import sys

np.random.seed(1)

def logsig(x):
    '''stabilised implementation taken from https://fa.bianp.net/blog/2019/evaluate_logistic/'''
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def j_theta(w, data_x, data_y):
    '''stabilised implementation taken from https://fa.bianp.net/blog/2019/evaluate_logistic/'''
    z = np.dot(data_x.T, w)
    return np.mean((1 - data_y) * z - logsig(z))

def j_theta_grad(w, data_x, data_y):
    '''stabilised implementation taken from https://fa.bianp.net/blog/2019/evaluate_logistic/'''
    return j_theta(w, data_x, data_y), gradient(w, data_x, data_y).ravel()
def expit_b(x, b):
    '''stabilised implementation taken from https://fa.bianp.net/blog/2019/evaluate_logistic/'''
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


def gradient(w, data_x, data_y):
    '''stabilised implementation taken from https://fa.bianp.net/blog/2019/evaluate_logistic/'''
    z = data_x.T.dot(w)
    s = expit_b(z, data_y)
    return data_x.dot(s) / data_x.shape[1]

def predict(w, data_x):
    y_hat = logistic(np.dot(w, data_x))
    return (y_hat > 0.5).astype(int)

def logistic(x):
    return 1/(1+np.exp(-1*x))

def get_probs(w, data_x):
    return logistic(np.dot(w, data_x))

def logistic_regression(data_x, data_y):
    w = -1 + 2*np.random.random(data_x.shape[0])
    tol = 1e-4
    max_iter = 1000
    opt_res = optimize.minimize(j_theta_grad, w, args=(data_x, data_y), method='L-BFGS-B', jac=True,
                                options={"disp": False, "gtol": tol, "maxiter": max_iter})
    w = opt_res.x
    return w

def lr_run(data_name):
    output_dict = {}
    output_dict['dataset'] = data_name
    output_dict['algo'] = 'lr'
    data_x, data_y, data_x_test, data_y_test = get_dataset(data_name)
    output_dict['train_examples'] = data_x.shape[0]
    output_dict['columns'] = data_x.shape[1]
    output_dict['test_examples'] = data_x_test.shape[0]
    output_dict['positive_train_examples'] = np.sum(data_y)
    output_dict['positive_test_examples'] = np.sum(data_y_test)
    output_dict['negative_train_examples'] = len(data_y) - np.sum(data_y)
    output_dict['negative_test_examples'] = len(data_y_test) - np.sum(data_y_test)
    data_x = data_x.T
    data_x_test = data_x_test.T
    data_x = np.vstack((data_x, np.ones(data_x.shape[1])))
    data_x_test = np.vstack((data_x_test, np.ones(data_x_test.shape[1])))
    clf = logistic_regression(data_x, data_y)
    predicted_labels = predict(clf, data_x)
    predicted_labels_test = predict(clf, data_x_test)

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
    roc_train = roc_auc_score(data_y, get_probs(clf, data_x))
    roc_test = roc_auc_score(data_y_test, get_probs(clf, data_x_test))
    output_dict['roc_train'] = roc_train
    output_dict['roc_test'] = roc_test
    print(output_dict)


lr_run(sys.argv[1])
