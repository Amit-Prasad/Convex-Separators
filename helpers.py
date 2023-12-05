import numpy as np
from sklearn.gaussian_process.kernels import RBF

def check_label(x, u):
    '''if x=u return 1, else return 0'''
    return np.where(x == u,1,0)

def random_false_positive_index(unused_fp):
    false_positive_indices = np.where(unused_fp == 1)[0]
    if false_positive_indices.size == 0:
        return -1
    return np.random.choice(false_positive_indices, 1, replace=True)

#Approach 1
def hyperplane_init_objective(w, data_x, data_y, x_fp):
    dot_prod = np.dot(w, data_x) - np.dot(w, x_fp)
    dot_prod[dot_prod>=1] = 1
    return np.sum(dot_prod * check_label(data_y, 1))

def gradient_init_hyperplanes(w, data_x, data_y, x_fp):
    pos_indices = np.where(data_y==1)[0]
    indices = np.where(np.dot(w[0:-1], data_x[0:-1, pos_indices]) - np.dot(w[0:-1], x_fp[0:-1]) < 1)[0]
    gradient = np.sum(data_x[:, pos_indices][:, indices], axis=1) - len(indices) * x_fp
    return gradient

def hyperplane_init_objective_lbfgs(w, data_x, data_y, x_fp):
    return hyperplane_init_objective(w[0:-1], data_x[0:-1,:], data_y, x_fp[0:-1])*-1, gradient_init_hyperplanes(w, data_x, data_y, x_fp)*-1


def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y.shape) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point.reshape(-1, 1), axis = 0)
    min_index = np.argmin(dist)
    return min_index

def sample_boundary(data_x, data_y, n_sample):
    data_s = np.copy(data_x)
    data_s_y = np.copy(data_y)
    candidate_points_pos = []
    candidate_points_neg = []
    for j in range(n_sample):
        p2_old = -1
        p3_old = -1
        index = np.random.choice(np.where(data_s_y == 0)[0], 1, replace=True)
        x_fp = data_s[:, index]
        while True:
            p2 = min_distance_point(x_fp, data_s, data_s_y, 1)
            p3 = min_distance_point(data_s[:, p2], data_s, data_s_y, 0)
            if p2 == p2_old and p3 == p3_old:
                break
            p2_old = p2
            p3_old = p3
            x_fp = data_s[:, p3]
        candidate_points_pos.append(data_s[:, p2_old])
        candidate_points_neg.append(data_s[:, p3_old])
        data_s = np.delete(data_s, [p2_old, p3_old], 1)
        data_s_y = np.delete(data_s_y, [p2_old, p3_old])
    candidate_points_pos = np.asarray(candidate_points_pos).T
    candidate_points_neg = np.asarray(candidate_points_neg).T
    return candidate_points_pos, candidate_points_neg

def sample_boundary_for_fp(data_x, data_y):
    data_s = data_x
    data_s_y = data_y
    p2_old = -1
    p3_old = -1
    index = np.random.choice(np.where(data_s_y == 0)[0], 1, replace=True)
    x_fp = data_s[:, index]
    while True:
        p2 = min_distance_point(x_fp, data_s, data_s_y, 1)
        p3 = min_distance_point(data_s[:, p2], data_s, data_s_y, 0)
        if p2 == p2_old and p3 == p3_old:
            break
        p2_old = p2
        p3_old = p3
        x_fp = data_s[:, p3]
    return p3_old

def feature_engineer_rbf(data_x, data_y, gamma=1):
    boundary_pos, boundary_neg = sample_boundary(data_x, data_y, 40)
    feature_pos = np.hstack((boundary_pos, boundary_neg))
    data_x = RBF(gamma).__call__(data_x.T, feature_pos.T)
    return data_x.T, feature_pos

def feature_engineer_rbf_sampled(data_x, feature_pos, gamma=1):
    data_x = RBF(gamma).__call__(data_x.T, feature_pos.T)
    return data_x.T, feature_pos

def feature_engineer_quadratic(data_x):
    qua = data_x**2
    pairwise = np.zeros(((data_x.shape[0] * (data_x.shape[0] - 1))//2, data_x.shape[1]))
    p = 0
    for i in range(0, data_x.shape[0]):
        for j in range(i+1, data_x.shape[0]):
            pairwise[p, :] = data_x[i, :] * data_x[j, :]
            p+=1
    data_new = np.zeros((data_x.shape[0] * 2 + (data_x.shape[0] * (data_x.shape[0] - 1))//2, data_x.shape[1]))
    data_new[0:data_x.shape[0], :] = data_x
    data_new[data_x.shape[0]:2*data_x.shape[0], :] = qua
    data_new[2*data_x.shape[0]:2*data_x.shape[0] + (data_x.shape[0] * (data_x.shape[0] - 1))//2, :] = pairwise
    return data_new

def feature_engineer_quadratic_fetch(data_x):
    qua = data_x**2
    pairwise = np.zeros((data_x.shape[0], (data_x.shape[1] * (data_x.shape[1] - 1))//2))
    p = 0
    for i in range(0, data_x.shape[1]):
        for j in range(i+1, data_x.shape[1]):
            pairwise[:, p] = data_x[:, i] * data_x[:, j]
            p+=1
    data_new = np.zeros((data_x.shape[0], data_x.shape[1] * 2 + (data_x.shape[1] * (data_x.shape[1] - 1))//2))
    data_new[:, 0:data_x.shape[1]] = data_x
    data_new[:, data_x.shape[1]:2*data_x.shape[1]] = qua
    data_new[:, 2*data_x.shape[1]:2*data_x.shape[1] + (data_x.shape[1] * (data_x.shape[1] - 1))//2] = pairwise
    min_col_wise = np.min(data_new, axis=0)
    max_col_wise = np.max(data_new, axis=0)
    invalid = np.where(min_col_wise == max_col_wise)[0]
    data_new = np.delete(data_new, invalid, 1)
    return data_new

def featurewise_normalisation_fetch(data_x):
    means = np.mean(data_x, axis=0)
    std = np.std(data_x, axis=0)
    data_x = (data_x - np.tile(means, (data_x.shape[0], 1)))/std
    return data_x

def featurewise_normalisation(data_x):
    means = np.mean(data_x, axis=1)
    std = np.std(data_x, axis=1)
    data_x = ((data_x-means.reshape(-1, 1)).T/std).T
    return data_x

def append_bias(data_x):
    data_x_temp = np.zeros((data_x.shape[0]+1, data_x.shape[1]))
    data_x_temp[0:-1, :] = data_x
    data_x_temp[-1, :] = np.ones(data_x.shape[1])
    return data_x_temp
