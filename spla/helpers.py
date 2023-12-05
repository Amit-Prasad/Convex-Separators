import numpy as np
import matplotlib.pyplot as plt

def check_label(x, u):
    '''if x=u return 1, else return 0'''
    return np.where(x == u,1,0)


def random_false_positive_index(labels, data_y):
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    if false_positive_indices.size == 0:
        return -1
    return np.random.choice(false_positive_indices, 1, replace=True)


#Approach 3
def hyperplane_init_objective(w, data_x, data_y, x_fp):
    dot_prod = np.dot(w, data_x) - np.dot(w, x_fp)
    dot_prod_1 = np.copy(dot_prod)
    dot_prod_2 = np.copy(dot_prod)
    dot_prod_1[dot_prod_1 >= 1] = 1
    dot_prod_2[dot_prod_2 >= 0] = 0
    return np.sum((dot_prod_1 + dot_prod_2) * check_label(data_y, 1))