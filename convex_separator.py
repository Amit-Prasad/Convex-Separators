import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
from fpdf import FPDF
from helpers import *
from scipy.special import logsumexp, softmax
from scipy import optimize
from sklearn.metrics import accuracy_score

color_map = {}
dir_points = {}
j=0
i=0
for k in range(0, 150):
    color_map[k] = cm.rainbow(k*0.01)
    if i*0.2 == 1.8:
        j += 0.1
        i=0
    dir_points[k] = [0.8 + j, i*0.2]
    i+=1



def add_hyperplane(W, data_x, data_y, pdf, boundary, first, visualise = 0):
    W_new = np.zeros((W.shape[0], W.shape[1] + 1))
    w, ter_code, data_x, data_y = learn_init_hyperplane(W, data_x, data_y, boundary, first, pdf, visualise)
    if w is None:
        return W, ter_code, data_x, data_y
    if first:
        return w.reshape(-1, 1), ter_code, data_x, data_y
    norms = np.linalg.norm(W, axis=0)
    scale = np.amin(norms) + np.random.random() * (np.amax(norms) - np.amin(norms))
    W_new[:, 0:-1] = W
    W_new[:, -1] = w * scale / (np.linalg.norm(w))
    return W_new, ter_code, data_x, data_y
def delete_dead_hyperplanes(W, data_x):
    i=0
    norms = np.linalg.norm(W[0:-1, :], axis=0)
    W = W[:, np.argsort(norms)]
    labels = predict(W, data_x)
    while i < W.shape[1]:
        W[:, i] = -1*W[:, i]
        labels1 = predict(W, data_x)
        if np.array_equal(labels, labels1):
            W = np.delete(W, i, axis=1)
        else:
            W[:, i] = -1*W[:, i]
            i+=1
    return W

def delete_dead_hyperplanes_cs1(W, data_x):
    i=0
    labels = predict(W, data_x)
    while i < W.shape[1]:
        W[:, i] = -1*W[:, i]
        labels1 = predict(W, data_x)
        if np.array_equal(labels, labels1):
            W = np.delete(W, i, axis=1)
        else:
            W[:, i] = -1*W[:, i]
            i+=1
    return W

#def y_i_hat(W, data_x):
#    y_i_hat = np.amin(np.dot(W.T, data_x), axis=0)
#    return y_i_hat

#def logistic(x):
#    return 1/(1+np.exp(-1*x))

#def h_w(W, data_x):
#    return logistic(y_i_hat(W, data_x))

def h_w_softmin(W, data_x):
    # return 1/(1+np.sum(np.exp(-1*np.dot(W.T, data_x)), axis=0))
    temp = -1 * np.dot(W.T, data_x)
    temp1 = np.vstack((np.zeros(data_x.shape[1]), temp))
    temp2 = logsumexp(temp1, axis=0)
    return np.exp(-1 * temp2)
def j_theta_softmin(W, data_x, data_y, balanced = False, regularised = 'None', c=0.5, epsilon = 0.001):
    # return np.mean(data_y*np.log(h_w(W, data_x)) + (1-data_y)*np.log(1 - h_w(W, data_x))) * (-1)
    temp = -1*np.dot(W.T, data_x)
    temp1 = np.vstack((np.zeros(data_x.shape[1]), temp))
    temp2 = logsumexp(temp1, axis=0)
    temp3 = logsumexp(temp, axis=0)
    loss = 0
    if balanced:
        weight_pos = 1/np.sum(data_y)
        weight_neg = 1/(len(data_y) - np.sum(data_y))
        loss += -1*np.sum(-1 *weight_pos*data_y*temp2 + weight_neg * (1-data_y)*temp3 - weight_neg * (1-data_y)*temp2)
    else:
        loss += -1*np.mean(-1*data_y*temp2 + (1-data_y)*temp3 - (1-data_y)*temp2)
    if regularised == 'l2':
        loss += c*np.sum(W[0:-1, :]**2)/data_x.shape[1]
    if regularised == 'l1':
        #loss += c*np.sum(np.sqrt(W[0:-1, :]**2 + epsilon**(1/(1+W[0:-1, :]**2))))/data_x.shape[1]
        loss += c*np.sum(np.sqrt(W[0:-1, :]**2 + epsilon))/data_x.shape[1]
    return loss

def gradient_softmin_stable(W, data_x, data_y, balanced = False, regularised = 'None', c=0.5, epsilon = 0.001):
    '''process positive and negative separately to avoid divide by zero'''
    temp = -1*np.dot(W.T, data_x)
    temp1 = np.vstack((np.zeros(data_x.shape[1]), temp))
    temp = softmax(temp1, axis=0)[1:, :]
    pos_indices = np.where(data_y==1)[0]
    neg_indices = np.where(data_y==0)[0]
    gradient = np.zeros(W.shape)
    weight_pos = 1/(len(data_y))
    weight_neg = 1/(len(data_y))
    if balanced:
        weight_pos = 1/np.sum(data_y)
        weight_neg = 1/(len(data_y) - np.sum(data_y))
    gradient += -1 * weight_pos * np.dot(data_x[:, pos_indices], temp[:, pos_indices].T)
    h_w_softmin_neg = h_w_softmin(W, data_x)[neg_indices]
    h_w_softmin_neg = np.where(h_w_softmin_neg > 0.999999999999999888977697537484, 0.999999999999999888977697537484, h_w_softmin_neg)  #replace by max floating point < 1 = 0.999999999999999888977697537484
    gradient += -1 * weight_neg * np.dot(np.tile(((-1 * h_w_softmin(W, data_x)[neg_indices])/(1-h_w_softmin_neg)), (data_x[:, neg_indices].shape[0], 1)) * data_x[:, neg_indices], temp[:, neg_indices].T)
    if regularised == 'l2':
        gradient[0:-1, :] += 2*c*W[0:-1, :]/data_x.shape[1]
    if regularised == 'l1':
        #additional_grad = (-2*W[0:-1, :]/((1+W[0:-1, :]**2)**2))*np.log(epsilon)*(epsilon**(1/(1+W[0:-1, :]**2)))
        #gradient[0:-1, :] += c*additional_grad/data_x.shape[1]
        additional_grad = W[0:-1, :]/np.sqrt(W[0:-1, :]**2 + epsilon)
        gradient[0:-1, :] += c*additional_grad/data_x.shape[1]
    return gradient

def j_theta_softmin_lbfgs(W, data_x, data_y, balanced = False, regularised = 'none', c=0.5, epsilon = 0.001):
    return j_theta_softmin(W.reshape(data_x.shape[0], -1), data_x, data_y, balanced, regularised, c, epsilon), gradient_softmin_stable(W.reshape(data_x.shape[0], -1), data_x, data_y, balanced, regularised, c, epsilon).ravel()

def bic(W, data_x, data_y, regularised = 'none'):
    if regularised == 'l1':
        if len(W.shape) > 1:
            error = 2 * j_theta_softmin(W, data_x, data_y, balanced=False, regularised='none') * data_x.shape[1]
            par_error = (np.sum(np.where(W[0:-1, :] != 0, 1, 0)) + W.shape[1]) * np.log(data_x.shape[1])
            return error, par_error, error + par_error
        error = 2 * j_theta_softmin(W, data_x, data_y, balanced=False, regularised='none') * data_x.shape[1]
        par_error = (np.sum(np.where(W[0:-1] != 0, 1, 0)) + 1) * np.log(data_x.shape[1])
        return error, par_error, error + par_error

    if len(W.shape) > 1:
        error = 2 * j_theta_softmin(W, data_x, data_y, balanced=False, regularised='none') * data_x.shape[1]
        par_error = (W.shape[0] * W.shape[1]) * np.log(data_x.shape[1])
        return error, par_error, error + par_error
    error = 2 * j_theta_softmin(W, data_x, data_y, balanced=False, regularised='none') * data_x.shape[1]
    par_error = (W.shape[0]) * np.log(data_x.shape[1])
    return error, par_error, error + par_error


def criteria_cs1(W, data_x, data_y):
    '''misclassified*d + n_hyp*(d+1) + c*h*(d+1)'''
    labels = predict(W, data_x)
    error = len(np.where(labels!=data_y)[0])
    par_error = W.shape[1]*(data_x.shape[0])
    return error, par_error, error + par_error


#def l_theta(W, data_x, data_y):
#    return (np.sum(data_y*np.log(h_w(W, data_x)) + (1-data_y)*np.log(1-h_w(W, data_x)))/data_x.shape[1])

#def j_theta(W, data_x, data_y, lamb):
#    return -1 * (np.sum(data_y * np.log(h_w(W.reshape(data_x.shape[0], -1), data_x)) + (1 - data_y) * np.log(1 - h_w(W.reshape(data_x.shape[0], -1), data_x)))/data_x.shape[1]) + lamb*(np.sum(np.abs(W)))

def j_theta_grad(W, data_x, data_y, lamb):
    return -1 * (np.sum(data_y * np.log(h_w(W.reshape(data_x.shape[0], -1), data_x)) + (1 - data_y) * np.log(1 - h_w(W.reshape(data_x.shape[0], -1), data_x)))/data_x.shape[1]) + lamb*(np.sum(np.abs(W))), gradient(W.reshape(data_x.shape[0], -1), data_x, data_y, lamb).ravel()

def soft_threshold_l1(W, lamb):
    W = np.sign(W)*np.where((np.abs(W) - lamb) > 0, (np.abs(W) - lamb), 0)
    return W

def learn_init_hyperplane(W, data_x, data_y, boundary, first, pdf, visualise = 0):
    '''termination codes: 0 implies no false positives left, 1 implies non convex separable, 2 implies succesfull in finding a hyperplane'''
    while True:
        labels = predict(W, data_x)
        w = -1 + 2 * np.random.random(W.shape[0])
        fps = np.zeros_like(data_y)
        fps[np.where(data_y==0)[0]] = 1
        if not first:
            fps = np.zeros_like(data_y)
            fps[np.intersect1d(np.where(data_y==0)[0], np.where(labels==1)[0])] = 1

        if boundary == True:
            pos_indices = np.where(data_y == 1)[0]
            false_pos_indices = np.where(fps == 1)[0]
            if len(false_pos_indices) == 0:
                fp_index = -1
            else:
                data_temp_x = np.copy(data_x)
                data_temp_y = 2*np.ones_like(data_y)
                data_temp_y[pos_indices] = 1
                data_temp_y[false_pos_indices] = 0
                fp_index = sample_boundary_for_fp(data_temp_x, data_temp_y)
        else:
            fp_index = random_false_positive_index(fps)
        if fp_index == -1:
            return None, 0, data_x, data_y
        x_fp = data_x[:, fp_index].reshape(-1, )
        w[-1] = -1 * np.dot(w[0:-1], x_fp[0:-1])
        epochs = 200
        learn_rate = 0.001
        tol = 1e-4
        count_k = 5
        k = count_k
        positive_indices = np.where(data_y == 1)[0]
        opt_res = optimize.minimize(hyperplane_init_objective_lbfgs, w, args=(data_x, data_y, x_fp), method='L-BFGS-B', jac=True,
                                    options={"disp": False, "gtol": tol, "maxiter": epochs})
        w = opt_res.x
        w[-1] = np.amax(-1 * np.dot(w[0:-1], data_x[0:-1, positive_indices]))
        if np.abs(np.sum(data_y)-hyperplane_init_objective(w[0:-1], data_x[0:-1, :], data_y, x_fp[0:-1]))>30:
            if visualise == 1:
                draw_init_lines(W, x_fp, w, data_x, data_y, 'rejected'+str(len(W)), 0, pdf)
                pdf.text(20, 436, 'rejected point = ' + str(x_fp))
                pdf.add_page()
            elif visualise == 2:
                draw_init_curves(W, x_fp, w, data_x, data_y, 'rejected'+str(len(W)), 0, pdf)
                pdf.text(20, 436, 'rejected point = ' + str(x_fp))
                pdf.add_page()
            data_x = np.delete(data_x, fp_index, 1)
            data_y = np.delete(data_y, fp_index)
            labels = predict(W, data_x)
            num_fp = len(np.intersect1d(np.where(data_y==0)[0], np.where(labels==1)[0]))
            if num_fp == 0:
                return None, 1, data_x, data_y
        else:
            if visualise == 1:
                draw_init_lines(W, x_fp, w, data_x, data_y, 'starts_tr_here'+str(len(W)), 0, pdf)
                pdf.add_page()
            if visualise == 2:
                draw_init_curves(W, x_fp, w, data_x, data_y, 'starts_tr_here'+str(len(W)), 0, pdf)
                pdf.add_page()
            return w, 2, data_x, data_y

def learn_init_k_hyperplanes(data_x, data_y, k, boundary, pdf, visualise=0):
    '''termination codes: 0 implies no negatives left, 1 implies non convex separable, 2 implies succesfull in finding a hyperplane'''
    count_hyp = 0
    W = np.zeros((data_x.shape[0], k))
    while count_hyp < k:
        w = -1 + 2 * np.random.random(data_x.shape[0])
        fps = np.zeros_like(data_y)
        fps[np.where(data_y == 0)[0]] = 1
        if boundary == True:
            pos_indices = np.where(data_y == 1)[0]
            false_pos_indices = np.where(fps == 1)[0]
            if len(false_pos_indices) == 0:
                fp_index = -1
            else:
                data_temp_x = np.copy(data_x)
                data_temp_y = 2 * np.ones_like(data_y)
                data_temp_y[pos_indices] = 1
                data_temp_y[false_pos_indices] = 0
                fp_index = sample_boundary_for_fp(data_temp_x, data_temp_y)
        else:
            fp_index = random_false_positive_index(fps)
        if fp_index == -1:
            return None, 0
        x_fp = data_x[:, fp_index].reshape(-1, )
        w[-1] = -1 * np.dot(w[0:-1], x_fp[0:-1])
        epochs = 200
        tol = 1e-4
        positive_indices = np.where(data_y == 1)[0]
        opt_res = optimize.minimize(hyperplane_init_objective_lbfgs, w, args=(data_x, data_y, x_fp), method='L-BFGS-B',
                                    jac=True, options={"disp": False, "gtol": tol, "maxiter": epochs})
        w = opt_res.x
        w[-1] = np.amax(-1 * np.dot(w[0:-1], data_x[0:-1, positive_indices]))
        if np.abs(np.sum(data_y) - hyperplane_init_objective(w[0:-1], data_x[0:-1, :], data_y, x_fp[0:-1])) > 30:
            if visualise == 1:
                draw_init_lines(w.reshape(-1, 1), x_fp, w, data_x, data_y, 'rejected' + str(len(w.reshape(-1, 1))), 0, pdf)
                pdf.text(20, 436, 'rejected point = ' + str(x_fp))
                pdf.add_page()
            elif visualise == 2:
                draw_init_curves(w.reshape(-1, 1), x_fp, w, data_x, data_y, 'rejected' + str(len(w.reshape(-1, 1))), 0, pdf)
                pdf.text(20, 436, 'rejected point = ' + str(x_fp))
                pdf.add_page()
            data_x = np.delete(data_x, fp_index, 1)
            data_y = np.delete(data_y, fp_index)
            num_fp = len(np.where(data_y == 0)[0])
            if num_fp == 0:
                return None, 1
        else:
            W[:, count_hyp] = w
            count_hyp += 1
            if visualise == 1:
                draw_init_lines(w.reshape(-1, 1), x_fp, w, data_x, data_y, 'starts_tr_here' + str(len(w.reshape(-1, 1))), 0, pdf)
                pdf.add_page()
            elif visualise == 2:
                draw_init_curves(w.reshape(-1, 1), x_fp, w, data_x, data_y, 'starts_tr_here' + str(len(w.reshape(-1, 1))), 0, pdf)
                pdf.add_page()
    return W/np.linalg.norm(W, axis=0), 2

def learn_hyperplanes(W, data_x, data_y, learn_rate, pdf, lamb, balanced=False, regularised='None', c=0.5, visualise = 0):
    if regularised!='l1':
        w = W.ravel()
        tol = 1e-4
        max_iter = 200
        opt_res = optimize.minimize(j_theta_softmin_lbfgs, w, args=(data_x, data_y, balanced, regularised, c), method='L-BFGS-B', jac = True, options={"disp": False, "gtol": tol, "maxiter": max_iter})
        w = opt_res.x
        W = w.reshape(data_x.shape[0], -1)
        if visualise == 1:
            draw_lines(W, data_x, data_y, 'ends_tr_here' + str(len(W)), learn_rate, 0, pdf, gradient_softmin_stable(W, data_x, data_y))
            pdf.add_page()
        elif visualise == 2:
            draw_curves(W, data_x, data_y, 'ends_tr_here' + str(len(W)), learn_rate, 0, pdf, gradient_softmin_stable(W, data_x, data_y))
            pdf.add_page()
    else:
        machine_epsilon = 1.40129846432e-45
        for i in range(3):
            epsilon = np.abs(W[0:-1, :])/1000 + machine_epsilon*10
            w = W.ravel()
            if i==0:
                epsilon = lamb*np.ones_like(W[0:-1, :])
            tol = 1e-4
            max_iter = 200
            if (W.shape[1] == 46) and ((i==1) or (i==0)):
                opt_res = optimize.minimize(j_theta_softmin_lbfgs, w, args=(data_x, data_y, balanced, regularised, lamb, epsilon), method='L-BFGS-B', jac=False, options={"disp": True, "gtol": tol, "maxiter": max_iter})
            else:
                opt_res = optimize.minimize(j_theta_softmin_lbfgs, w,
                                            args=(data_x, data_y, balanced, regularised, c, epsilon), method='L-BFGS-B',
                                            jac=True, options={"disp": False, "gtol": tol, "maxiter": max_iter})
            w = opt_res.x
            W = w.reshape(data_x.shape[0], -1)
            if visualise == True:
                draw_lines(W, data_x, data_y, 'ends_tr_here' + str(len(W)), learn_rate, 0, pdf, gradient_softmin_stable(W, data_x, data_y))
                pdf.add_page()
        W[0:-1, :] = np.where(np.abs(W[0:-1, :]) < lamb, 0, W[0:-1, :])
    return W

def predict(W, data_x):
    predict_pos = np.amin(np.dot(W.T, data_x), axis=0)
    prediction = np.where(predict_pos>0, 1, 0)
    return prediction

def get_probs(W, data_x):
    return h_w_softmin(W, data_x)


def draw_lines(W, data_x, data_y, iter, learn_rate, paper, pdf, derivative):
    #dir_points = {0: [1.5, 0], 1: [1.5, 0.2], 2: [1.5, 0.4], 3: [1.5, 0.6], 4: [1.5, 0.8], 5: [1.5, 1.0]}

    # plt.scatter(data_x[:, 0], data_x[:, 1], c=colors)
    # color_map = {0: 'cyan', 1: 'orange', 2: 'green', 3: 'black', 4: 'magenta', 5: 'red', 6: 'red', 7: 'red', 8: 'red', 9: 'red', 10: 'red'}
    labels = predict(W, data_x)
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))

    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='tab:blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='tab:orange', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='magenta', s=1)

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    grain = 0.001
    x1 = np.arange(x_min, x_max, grain)
    obj = j_theta_softmin(W, data_x, data_y)
    pdf.text(20, 436, 'learn rate = ' + str(learn_rate))
    pdf.text(20, 452, 'Objective = ' + str(obj))
    k=0
    for i in range(0, W.shape[1]):
        w = W[:, i]
        pdf.text(20, 452 + 14*(k+1), 'hyperplane_' + str(w))
        pdf.text(20, 452 + 14*(k+2), 'hyperplane_derivative' + str(derivative[:, i]))
        k+=2
        if w[1] == 0:
            x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
        else:
            x2 = -1 * (w[2] + w[0] * x1) / w[1]
        plt.plot(x1, x2, color='brown', linewidth=1)
        sgn = (w[0] * dir_points[i][0] + w[1] * dir_points[i][1] + w[2])
        # if sgn > 0:
        #     plt.scatter(dir_points[i][0], dir_points[i][1], marker='+',
        #                 color=color_map[i])
        # if sgn < 0:
        #     plt.scatter(dir_points[i][0], dir_points[i][1], marker='o',
        #                 color=color_map[i])
        plt.axis([x_min, x_max, y_min, y_max])
    if paper:
        plt.savefig('hyp_' + str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png', dpi=300)
    else:
        plt.savefig('hyp_'+ str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png')
    pdf.image('hyp_' + str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png', x=0, y=0, w=700, h=420)
    plt.gca().cla()

def draw_curves(W, data_x, data_y, iter, learn_rate, paper, pdf, derivative):
    #dir_points = {0: [1.5, 0], 1: [1.5, 0.2], 2: [1.5, 0.4], 3: [1.5, 0.6], 4: [1.5, 0.8], 5: [1.5, 1.0]}

    # plt.scatter(data_x[:, 0], data_x[:, 1], c=colors)
    #color_map = {0: 'cyan', 1: 'orange', 2: 'green', 3: 'black', 4: 'magenta', 5: 'red', 6:'pink'}
    labels = predict(W, data_x)
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))

    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='tab:blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='tab:orange', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='magenta', s=1)

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    grain = 0.001
    x1 = np.arange(x_min, x_max, grain)
    obj = j_theta_softmin(W, data_x, data_y)
    pdf.text(20, 436, 'learn rate = ' + str(learn_rate))
    pdf.text(20, 452, 'Objective = ' + str(obj))
    pdf.text(20, 468, 'number of hyperplanes = ' + str(W.shape[1]))
    k=0
    for i in range(0, W.shape[1]):
        w = W[:, i]
        pdf.text(20, 468 + 14*(k+1), 'hyperplane_' + str(w))
        pdf.text(20, 468 + 14*(k+2), 'hyperplane_derivative' + str(derivative[:, i]))
        k+=2
        discriminant = (w[1] + w[4] * x1)**2 - 4 * w[3] * (w[0] * x1 + w[2] * (x1**2) + w[5])
        pos_indices = np.where(discriminant>=0)[0]
        discriminant = discriminant[pos_indices]
        if discriminant.size == 0:
            continue
        b_term = -1*(w[1] + w[4] * x1)[pos_indices]
        x2 = (b_term + np.sqrt(discriminant))/(2*w[3])
        x3 = (b_term - np.sqrt(discriminant)) / (2 * w[3])
        plt.plot(x1[pos_indices], x2, color='brown')
        plt.plot(x1[pos_indices], x3, color='brown')
        plt.axis([x_min, x_max, y_min, y_max])
    if paper:
        plt.savefig('hyp_'+ str(W.shape[1]) + str(iter) + str(W[0,0]) + '.png', dpi=300)
    else:
        plt.savefig('hyp_' + str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png')
    pdf.image('hyp_' + str(W.shape[1]) + str(iter) + str(W[0,0]) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()


def draw_init_lines(W, x_fp, w_init, data_x, data_y, iter, paper, pdf):
    labels = predict(W, data_x)
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))

    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='tab:blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='tab:orange', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='magenta', s=1)
    plt.scatter(x_fp[0], x_fp[1], marker='+', color='black')

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    grain = 0.001
    x1 = np.arange(x_min, x_max, grain)
    k=0

    for i in range(0, W.shape[1]):
        w = W[:, i]
        if w[1] == 0:
            x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
        else:
            x2 = -1 * (w[2] + w[0] * x1) / w[1]
        plt.plot(x1, x2, color='brown', linewidth=1)
        sgn = (w[0] * dir_points[i][0] + w[1] * dir_points[i][1] + w[2])
        # if sgn > 0:
        #     plt.scatter(dir_points[i][0], dir_points[i][1], marker='+',
        #                 color=color_map[i])
        # if sgn < 0:
        #     plt.scatter(dir_points[i][0], dir_points[i][1], marker='o',
        #                 color=color_map[i])
        plt.axis([x_min, x_max, y_min, y_max])

    i = W.shape[1]
    w = w_init
    if w[1] == 0:
        x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
    else:
        x2 = -1 * (w[2] + w[0] * x1) / w[1]
    plt.plot(x1, x2, color='black', linewidth = 1, linestyle = 'dashed')
    sgn = (w[0] * dir_points[i][0] + w[1] * dir_points[i][1] + w[2])
    # if sgn > 0:
    #     plt.scatter(dir_points[i][0], dir_points[i][1], marker='+',
    #                 color='black')
    # if sgn < 0:
    #     plt.scatter(dir_points[i][0], dir_points[i][1], marker='o',
    #                 color='black')
    # plt.axis([x_min, x_max, y_min, y_max])

    if paper:
        plt.savefig('hyp_init'+ str(w_init[0]) + str(iter) + '.png', dpi = 300)
    else:
        plt.savefig('hyp_init' + str(w_init[0]) + str(iter) + '.png')
    pdf.image('hyp_init' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()

def draw_init_curves(W, x_fp, w_init, data_x, data_y, iter, paper, pdf):
    labels = predict(W, data_x)
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))
    plt.scatter(x_fp[0], x_fp[1], marker='+', color='black')
    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='tab:blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='tab:orange', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='magenta', s=1)

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    grain = 0.001
    x1 = np.arange(x_min, x_max, grain)
    for i in range(0, W.shape[1]):
        w = W[:, i]
        discriminant = (w[1] + w[4] * x1) ** 2 - 4 * w[3] * (w[0] * x1 + w[2] * (x1 ** 2) + w[5])
        pos_indices = np.where(discriminant >= 0)[0]
        discriminant = discriminant[pos_indices]
        if discriminant.size == 0:
            continue
        b_term = -1 * (w[1] + w[4] * x1)[pos_indices]
        x2 = (b_term + np.sqrt(discriminant)) / (2 * w[3])
        x3 = (b_term - np.sqrt(discriminant)) / (2 * w[3])
        plt.plot(x1[pos_indices], x2, color=color_map[i])
        plt.plot(x1[pos_indices], x3, color=color_map[149 - i])
        plt.axis([x_min, x_max, y_min, y_max])

    i = W.shape[1]
    w = w_init
    discriminant = (w[1] + w[4] * x1) ** 2 - 4 * w[3] * (w[0] * x1 + w[2] * (x1 ** 2) + w[5])
    pos_indices = np.where(discriminant >= 0)[0]
    discriminant = discriminant[pos_indices]
    if discriminant.size != 0:
        b_term = -1 * (w[1] + w[4] * x1)[pos_indices]
        x2 = (b_term + np.sqrt(discriminant)) / (2 * w[3])
        x3 = (b_term - np.sqrt(discriminant)) / (2 * w[3])
        plt.plot(x1[pos_indices], x2, color=color_map[i])
        plt.plot(x1[pos_indices], x3, color=color_map[149 - i])
        plt.axis([x_min, x_max, y_min, y_max])

    if paper:
        plt.savefig('hyp_init' + str(w_init[0]) + str(iter) + '.png', dpi=300)
    else:
        plt.savefig('hyp_init' + str(w_init[0]) + str(iter) + '.png')
    pdf.image('hyp_init' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700, h=420)
    plt.gca().cla()

def train_cs1(data_x, data_y, val_x, val_y, n_hyp, out_file, boundary, lamb, cri_enable, val_mode, error_threshold, visualise = 0):
    first = True
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'log')) is False:
        os.makedirs(os.path.join(current_dir, 'log'))
    os.chdir(os.path.join(current_dir, 'log'))
    W = np.random.rand(data_x.shape[0], 1)
    cri_prev = 1e6
    val_prev = -1
    miss_prev = max(len(data_y) - np.sum(data_y), np.sum(data_y))
    miss_prev_1 = 0
    ter_code = 0

    all_error = []
    all_error_val = []
    all_par_error = []
    all_vals = []
    all_acc = []
    all_acc_vals = []
    for i in range(0, n_hyp):
        W_prev = np.copy(W)
        # val = 0 (no stop, no delete, no bic stop), val=1 (no stop, no delete, bic stop), val=2 (no stop, delete, no bic stop)
        # val = 3 (no stop, delete, bic stop), val=4 (stop, no delete, no bic stop), val=5 (stop, no delete, bic stop)
        # val = 6 (stop, delete, no bic stop), val=7 (stop, delete, bic stop)
        W, temp_ter_code = add_hyperplane(W, data_x, data_y, pdf, boundary, first, visualise)
        error, par_error, cri_new = criteria_cs1(W, data_x, data_y)
        error_val, _, _ = criteria_cs1(W, val_x, val_y)
        miss_new = error
        pre_labels = predict(W, data_x)
        pre_val_labels = predict(W, val_x)
        val_new = accuracy_score(val_y, pre_val_labels)
        all_acc_vals.append(val_new)
        all_acc.append(accuracy_score(data_y, pre_labels))
        all_error.append(error)
        all_par_error.append(par_error)
        all_vals.append(cri_new)
        all_error_val.append(error_val)
        if (val_prev > val_new) and (val_mode is True):
            return W, 10, all_error, all_error_val, all_par_error, all_vals, all_acc, all_acc_vals, (miss_prev_1 - miss_prev)
        val_prev = val_new
        if (miss_prev - miss_new)<error_threshold and (val_mode is False):
            return W, 10, all_error, all_error_val, all_par_error, all_vals, all_acc, all_acc_vals, 0
        if temp_ter_code == 1:
            return W_prev, 7, all_error, all_error_val, all_par_error, all_vals, all_acc, all_acc_vals, (miss_prev_1 - miss_prev)
        elif temp_ter_code == 0:
            ter_code = 4
            W = W_prev
            break
        elif not first and cri_enable:
            error, par_error, cri_new = criteria_cs1(W, data_x, data_y)
            if cri_new > cri_prev:
                W = W_prev
                ter_code = 1
                break
            cri_prev = cri_new
        first = False
        miss_prev_1 = miss_prev
        miss_prev = miss_new
    size_prior_delete = W.shape[1]
    W = delete_dead_hyperplanes_cs1(W, data_x)
    size_post_delete = W.shape[1]
    if size_post_delete < size_prior_delete:
        if ter_code == 0:
            ter_code = 2
        elif ter_code == 4:
            ter_code = 6
        elif ter_code == 1:
            ter_code = 3
    if visualise:
        pdf.output(out_file + '.pdf')
    os.chdir(os.path.join(current_dir))
    return W, ter_code, all_error, all_error_val, all_par_error, all_vals, all_acc, all_acc_vals, (miss_prev_1 - miss_prev)


def train_cs2(data_x, data_y, n_hyp, out_file, boundary, lamb2, balanced = False, regularised = 'None', c=0.5, bic_enable = False, visualise = 0):
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'log')) is False:
        os.makedirs(os.path.join(current_dir, 'log'))
    os.chdir(os.path.join(current_dir, 'log'))
    all_j_theta = []
    all_par_error = []
    all_bic_vals = []
    all_acc = []
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    unused_fp = 1 - data_y
    W_one = -1 + 2 * np.random.random(data_x.shape[0]).reshape(-1, 1)
    W_one = learn_hyperplanes(W_one, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
    n_hyp = n_hyp - 1
    j_theta_val, par_error, bic_prev = bic(W_one, data_x, data_y, regularised)
    all_j_theta.append(j_theta_val)
    all_par_error.append(par_error)
    all_bic_vals.append(bic_prev)
    pre_labels = predict(W_one, data_x)
    all_acc.append(accuracy_score(data_y, pre_labels))
    if n_hyp==0:
        if visualise == 1 or visualise == 2:
            pdf.output(out_file + '.pdf')
        return W_one, 0, all_j_theta, all_par_error, all_bic_vals
    W, ter_code = learn_init_k_hyperplanes(data_x, data_y, 2, boundary, pdf, visualise)
    if ter_code == 0 or ter_code == 1:
        if visualise == 1 or visualise == 2:
            pdf.output(out_file + '.pdf')
        return W_one, 7, all_j_theta, all_par_error, all_bic_vals, all_acc
    ter_code = 0
    first = False
    W = learn_hyperplanes(W, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
    n_hyp = n_hyp - 1
    j_theta_val, par_error, bic_new = bic(W, data_x, data_y, regularised)
    all_j_theta.append(j_theta_val)
    all_par_error.append(par_error)
    all_bic_vals.append(bic_new)
    pre_labels = predict(W, data_x)
    all_acc.append(accuracy_score(data_y, pre_labels))
    if bic_new > bic_prev and bic_enable == True:
        ter_code = 1
        if visualise == 1 or visualise == 2:
            pdf.output(out_file + '.pdf')
        return W_one, ter_code, all_j_theta, all_par_error, all_bic_vals, all_acc
    bic_prev = bic_new
    # val = 0 (no stop, no delete, no bic stop), val=1 (no stop, no delete, bic stop), val=2 (no stop, delete, no bic stop)
    # val = 3 (no stop, delete, bic stop), val=4 (stop, no delete, no bic stop), val=5 (stop, no delete, bic stop)
    # val = 6 (stop, delete, no bic stop), val=7 (stop, delete, bic stop), val = 7 (not convex separable)
    for i in range(0, n_hyp):
        start_size = W.shape[1]
        W_prev = np.copy(W)
        W, temp_ter_code, data_x, data_y = add_hyperplane(W, data_x, data_y, pdf, boundary, first, visualise)
        if temp_ter_code == 1:
            if visualise == 1 or visualise == 2:
                pdf.output(out_file + '.pdf')
            return W, 7, all_j_theta, all_par_error, all_bic_vals, all_acc
        end_size = W.shape[1]
        if start_size == end_size:
            ter_code = 4
            break
        W = learn_hyperplanes(W, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
        j_theta_val, par_error, bic_new = bic(W, data_x, data_y, regularised)
        all_j_theta.append(j_theta_val)
        all_par_error.append(par_error)
        all_bic_vals.append(bic_new)
        pre_labels = predict(W, data_x)
        all_acc.append(accuracy_score(data_y, pre_labels))
        if bic_new > bic_prev and bic_enable == True:
            ter_code = 1
            W = W_prev
            break
        bic_prev = bic_new
    size_prior_delete = W.shape[1]
    W = delete_dead_hyperplanes(W, data_x)
    size_post_delete = W.shape[1]
    if size_post_delete < size_prior_delete:
        if ter_code == 0:
            ter_code = 2
        elif ter_code == 4:
            ter_code = 6
        elif ter_code == 1:
            ter_code = 3
    if visualise == 1 or visualise == 2:
        pdf.output(out_file + '.pdf')
    os.chdir(os.path.join(current_dir))
    return W, ter_code, all_j_theta, all_par_error, all_bic_vals, all_acc


def train_cs2_k_hyperplane_search(data_x, data_y, n_hyp, out_file, boundary, lamb2, balanced = False, regularised = 'None', c=0.5, bic_enable = False, visualise = 0):
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'log')) is False:
        os.makedirs(os.path.join(current_dir, 'log'))
    os.chdir(os.path.join(current_dir, 'log'))
    all_j_theta = []
    all_par_error = []
    all_bic_vals = []
    all_acc = []
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    unused_fp = 1 - data_y
    W_one = -1 + 2 * np.random.random(data_x.shape[0]).reshape(-1, 1)
    W_one = learn_hyperplanes(W_one, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
    n_hyp = n_hyp - 1
    j_theta_val, par_error, bic_prev = bic(W_one, data_x, data_y, regularised)
    all_j_theta.append(j_theta_val)
    all_par_error.append(par_error)
    all_bic_vals.append(bic_prev)
    pre_labels = predict(W_one, data_x)
    all_acc.append(accuracy_score(data_y, pre_labels))
    if n_hyp==0:
        if visualise == 1 or visualise == 2:
            pdf.output(out_file + '.pdf')
        return W_one, 0, all_j_theta, all_par_error, all_bic_vals
    W, ter_code = learn_init_k_hyperplanes(data_x, data_y, 2, boundary, pdf, visualise)
    if ter_code == 0 or ter_code == 1:
        if visualise == 1 or visualise == 2:
            pdf.output(out_file + '.pdf')
        return W_one, 7, all_j_theta, all_par_error, all_bic_vals, all_acc
    ter_code = 0
    first = False
    W = learn_hyperplanes(W, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
    n_hyp = n_hyp - 1
    j_theta_val, par_error, bic_new = bic(W, data_x, data_y, regularised)
    all_j_theta.append(j_theta_val)
    all_par_error.append(par_error)
    all_bic_vals.append(bic_new)
    pre_labels = predict(W, data_x)
    all_acc.append(accuracy_score(data_y, pre_labels))
    if bic_new > bic_prev and bic_enable == True:
        found_better = False
        for k in range(5):
            W = np.delete(W, -1, 1)
            all_j_theta.pop()
            all_par_error.pop()
            all_bic_vals.pop()
            W, temp_ter_code, data_x, data_y = add_hyperplane(W, data_x, data_y, pdf, boundary, first, visualise)
            if temp_ter_code == 1 or temp_ter_code == 0:
                break
            W = learn_hyperplanes(W, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
            j_theta_val, par_error, bic_new = bic(W, data_x, data_y, regularised)
            all_j_theta.append(j_theta_val)
            all_par_error.append(par_error)
            all_bic_vals.append(bic_new)
            if bic_new < bic_prev:
                found_better = True
                break
        if found_better is False:
            ter_code = 1
            if visualise == 1 or visualise == 2:
                pdf.output(out_file + '.pdf')
            return W_one, ter_code, all_j_theta, all_par_error, all_bic_vals, all_acc
    bic_prev = bic_new
    # val = 0 (no stop, no delete, no bic stop), val=1 (no stop, no delete, bic stop), val=2 (no stop, delete, no bic stop)
    # val = 3 (no stop, delete, bic stop), val=4 (stop, no delete, no bic stop), val=5 (stop, no delete, bic stop)
    # val = 6 (stop, delete, no bic stop), val=7 (stop, delete, bic stop), val = 7 (not convex separable)
    for i in range(0, n_hyp):
        start_size = W.shape[1]
        W_prev = np.copy(W)
        W, temp_ter_code = add_hyperplane(W, data_x, data_y, pdf, boundary, first, visualise)
        if temp_ter_code == 1:
            if visualise == 1 or visualise == 2:
                pdf.output(out_file + '.pdf')
            return W, 7, all_j_theta, all_par_error, all_bic_vals, all_acc
        end_size = W.shape[1]
        if start_size == end_size:
            ter_code = 4
            break
        W = learn_hyperplanes(W, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
        j_theta_val, par_error, bic_new = bic(W, data_x, data_y, regularised)
        all_j_theta.append(j_theta_val)
        all_par_error.append(par_error)
        all_bic_vals.append(bic_new)
        pre_labels = predict(W, data_x)
        all_acc.append(accuracy_score(data_y, pre_labels))
        if bic_new > bic_prev and bic_enable == True:
            found_better = False
            for k in range(5):
                W = np.delete(W, -1, 1)
                all_j_theta.pop()
                all_par_error.pop()
                all_bic_vals.pop()
                W, temp_ter_code = add_hyperplane(W, data_x, data_y, pdf, boundary, first, visualise)
                if temp_ter_code == 1 or temp_ter_code == 0:
                    break
                W = learn_hyperplanes(W, data_x, data_y, 0.01, pdf, lamb2, balanced, regularised, c, visualise)
                j_theta_val, par_error, bic_new = bic(W, data_x, data_y, regularised)
                all_j_theta.append(j_theta_val)
                all_par_error.append(par_error)
                all_bic_vals.append(bic_new)
                if bic_new < bic_prev:
                    found_better = True
                    break
            if found_better is False:
                ter_code = 1
                W = W_prev
                break
        bic_prev = bic_new
    size_prior_delete = W.shape[1]
    W = delete_dead_hyperplanes(W, data_x)
    size_post_delete = W.shape[1]
    if size_post_delete < size_prior_delete:
        if ter_code == 0:
            ter_code = 2
        elif ter_code == 4:
            ter_code = 6
        elif ter_code == 1:
            ter_code = 3
    if visualise == 1 or visualise == 2:
        pdf.output(out_file + '.pdf')
    os.chdir(os.path.join(current_dir))
    return W, ter_code, all_j_theta, all_par_error, all_bic_vals, all_acc

def test(W, data_x):
    labels = predict(W, data_x)
    return labels
