from matplotlib.pyplot import cm
from fpdf import FPDF
from helpers import *
from scipy import optimize

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

def j_theta_softmin_lbfgs(W, data_x, data_y):
    return (-1/data_x.shape[1])*l_theta(W.reshape(data_x.shape[0], 1), data_x, data_y), (-1/data_x.shape[1])*gradient(W.reshape(data_x.shape[0], 1), data_x, data_y).ravel()

def logistic_regression(data_x, data_y, pdf, visualise=False):
    w = -1 + 2*np.random.random(data_x.shape[0])
    tol = 1e-4
    max_iter = 200
    opt_res = optimize.minimize(j_theta_softmin_lbfgs, w, args=(data_x, data_y), method='L-BFGS-B', jac=True,
                                options={"disp": False, "gtol": tol, "maxiter": max_iter})
    w = opt_res.x
    W = w.reshape(data_x.shape[0], -1)
    if visualise == True:
        draw_lines(W, data_x, data_y, 'ends_tr_here' + str(len(W)), 0, 0, pdf, (-1/data_x.shape[1])*gradient(W, data_x, data_y))
        pdf.add_page()
    return W

def logistic_regression_k(data_x, data_y, k, pdf, visualise=False):
    W = np.zeros((data_x.shape[0], k))
    tol = 1e-4
    max_iter = 200
    # choose random attribute to sort and then do k_parts
    random_axis = np.random.choice(np.arange(data_x.shape[0] - 1), 1, replace=True)
    attributes = data_x[random_axis, :]
    sorted_indices = np.argsort(attributes)[0]
    data_x_temp = data_x[:, sorted_indices]
    data_y_temp = data_y[sorted_indices]
    batch_size = data_x.shape[1] // k
    for i in range(k):
        w = -1 + 2*np.random.random(data_x.shape[0])
        if i == k-1:
            data_x_pre = data_x_temp[:, i * batch_size:]
            data_y_pre = data_y_temp[i * batch_size:]
        else:
            data_x_pre = data_x_temp[:, i * batch_size:(i + 1) * batch_size]
            data_y_pre = data_y_temp[i * batch_size:(i + 1) * batch_size]
        opt_res = optimize.minimize(j_theta_softmin_lbfgs, w, args=(data_x_pre, data_y_pre), method='L-BFGS-B', jac=True,
                                    options={"disp": False, "gtol": tol, "maxiter": max_iter})

        W[:, i] = opt_res.x
        w = opt_res.x
        if visualise == True:
            draw_lines(w.reshape(-1, 1), data_x_pre, data_y_pre, 'ends_tr_here' + str(len(W)), 0, 0, pdf, (-1/data_x.shape[1])*gradient(W, data_x, data_y))
            pdf.add_page()
    return W

def bic(W, data_x, data_y):
    if len(W.shape)>1:
        error = -2 * l_theta(W, data_x, data_y)
        par_error = (W.shape[0] * W.shape[1]) * np.log(data_x.shape[1])
        return error, par_error, error + par_error
    error = -2 * l_theta(W, data_x, data_y)
    par_error = (W.shape[0]) * np.log(data_x.shape[1])
    return error, par_error, error + par_error

def indicator_min(x):
    return (x <= np.sort(x, axis=0)[::-1][[-1], :]).astype(int)

def indicator_max(x):
    return (x >= np.sort(x, axis=0)[[-1], :]).astype(int)

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

def logistic_hess(W, data_x):
    y_i_hat = np.amin(np.dot(W.T, data_x), axis=0)
    min_index = np.argmin(np.dot(W.T, data_x), axis=0)
    return 1/(1+np.exp(-1*y_i_hat)), min_index
def calc_hess(W, data_x):
    H = []
    sig, index = logistic_hess(W, data_x)
    a = sig*(1-sig)
    for i in range(0, W.shape[1]):
        index_temp = np.where(index == i)[0]
        if len(index_temp) == 0:
            H.append(np.identity(data_x.shape[0]))
            continue
        data_temp = data_x[:, index_temp]
        a_temp = a[index_temp]
        D = np.diag(a_temp.reshape(-1, 1).T[0])
        H.append(data_temp @ D @ data_temp.T)
    return H

def gradient(W, data_x, data_y, lamb = 0.01):
    '''vectorised, -1 in case of maximisation '''
    min_hyp_indices = np.argmin(np.dot(W.T, data_x), axis=0)
    min_x_indices = np.arange(0, data_x.shape[1])
    z = np.amin(np.dot(W.T, data_x), axis=0)
    s = expit_b(z, data_y)
    mat = np.zeros((data_x.shape[1], W.shape[1]))
    mat[min_x_indices, min_hyp_indices, ] = 1
    derivative = np.dot(np.tile(s, (data_x.shape[0], 1)) * data_x, mat)
    return derivative*-1

def l_theta(W, data_x, data_y):
    '''stabilised implementation taken from https://fa.bianp.net/blog/2019/evaluate_logistic/'''
    '''-1 in case of maximisation'''
    z = np.amin(np.dot(W.T, data_x), axis=0)
    return np.sum((1 - data_y) * z - logsig(z)) * (-1)

def y_i_hat(W, data_x):
    y_i_hat = np.amin(np.dot(W.T, data_x), axis=0)
    return y_i_hat

def get_probs(W, data_x):
    return h_w(W, data_x)

def logistic(x):
    return 1/(1+np.exp(-1*x))

def h_w(W, data_x):
    return logistic(y_i_hat(W, data_x))


def learn_hyperplanes(W, data_x, data_y, learn_rate, pdf, visualise = False):
    iter = 0
    epochs = 200
    count_k = 10
    k=10
    while iter<epochs:
        obj_start = l_theta(W, data_x, data_y)
        derivative = gradient(W, data_x, data_y)
        if iter%1 == 0 and visualise == True:
            if (W.shape[1] == 3 and iter == 199) or (W.shape[1] == 4 and iter == 0) or (W.shape[1] == 4 and iter == 199) or (W.shape[1] == 4 and iter == 2) or (W.shape[1] == 4 and iter == 4) or (W.shape[1] == 4 and iter == 7) or (W.shape[1] == 4 and iter == 10):
                draw_lines(W, data_x, data_y, iter, learn_rate, 1, pdf, derivative)
            else:
                draw_lines(W, data_x, data_y, iter, learn_rate, 0, pdf, derivative)
            pdf.add_page()
        W_new = W + learn_rate*derivative
        obj_end = l_theta(W_new, data_x, data_y)
        if(obj_end>obj_start):
            W = W_new
            k = k - 1
            if k <= 0:
                k = count_k
                learn_rate = learn_rate * 1.5
        else:
            k = count_k
            learn_rate = learn_rate / 2
        iter = iter + 1
    #draw_lines(W, data_x, data_y, 'ends_tr_here' + str(len(W)), learn_rate, 1, pdf, derivative)
    pdf.add_page()
    return W

def learn_hyperplanes_hess(W, data_x, data_y, learn_rate, pdf, visualise = False):
    iter = 0
    epochs = 20
    count_k = 10
    k=10
    while iter<epochs:
        obj_start = l_theta(W, data_x, data_y)
        derivative = gradient(W, data_x, data_y)
        if iter%1 == 0 and visualise == True:
            if (W.shape[1] == 3 and iter == 199) or (W.shape[1] == 4 and iter == 0) or (W.shape[1] == 4 and iter == 199) or (W.shape[1] == 4 and iter == 2) or (W.shape[1] == 4 and iter == 4) or (W.shape[1] == 4 and iter == 7) or (W.shape[1] == 4 and iter == 10):
                draw_lines(W, data_x, data_y, iter, learn_rate, 1, pdf, derivative)
            else:
                draw_lines(W, data_x, data_y, iter, learn_rate, 0, pdf, derivative)
            pdf.add_page()
        W_new = np.zeros_like(W)
        hessian = calc_hess(W, data_x)
        for i in range(0, W.shape[1]):
            W_new[:, i] = W[:, i] + np.dot(np.linalg.inv(hessian[i]), derivative[:, i])
        obj_end = l_theta(W_new, data_x, data_y)
        if(obj_end>obj_start):
            W = W_new
            k = k - 1
            if k <= 0:
                k = count_k
                learn_rate = learn_rate * 1.5
        else:
            k = count_k
            learn_rate = learn_rate / 2
        iter = iter + 1
    #draw_lines(W, data_x, data_y, 'ends_tr_here' + str(len(W)), learn_rate, 1, pdf, derivative)
    pdf.add_page()
    return W

def test(W, data_x):
    return predict(W, data_x)

def predict(W, data_x):
    predict_pos = np.amin(np.dot(W.T, data_x), axis=0)
    prediction = np.where(predict_pos>0, 1, 0)
    return prediction

def draw_lines(W, data_x, data_y, iter, learn_rate, paper, pdf, derivative):
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
    obj = l_theta(W, data_x, data_y)
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
        plt.axis([x_min, x_max, y_min, y_max])
    if paper:
        plt.savefig('paper_hyp_' + str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png', dpi=300)
        pdf.image('paper_hyp_' + str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png', x=0, y=0, w=700, h=420)
    else:
        plt.savefig('hyp_'+ str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png')
        pdf.image('hyp_' + str(W.shape[1]) + str(iter) + str(W[0, 0]) + '.png', x=0, y=0, w=700,h=420)
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
    obj = l_theta(W, data_x, data_y)
    k=0

    for i in range(0, W.shape[1]):
        w = W[:, i]
        if w[1] == 0:
            x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
        else:
            x2 = -1 * (w[2] + w[0] * x1) / w[1]
        plt.plot(x1, x2, color='brown', linewidth=1)
        plt.axis([x_min, x_max, y_min, y_max])

    i = W.shape[1]
    w = w_init
    if w[1] == 0:
        x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
    else:
        x2 = -1 * (w[2] + w[0] * x1) / w[1]
    plt.plot(x1, x2, color='black', linewidth = 1, linestyle = 'dashed')
    if paper:
        plt.savefig('paper_part'+ str(w_init[0]) + str(iter) + '.png', dpi = 300)
    else:
        plt.savefig('hyp_init' + str(w_init[0]) + str(iter) + '.png')
    pdf.image('hyp_init' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()

def train(data_x, data_y, max_hyp, visualise = False):
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    W_prev = logistic_regression(data_x, data_y, pdf)
    bic_prev = bic(W_prev, data_x, data_y)
    k=2
    ter_code=0 #1 bic termination, 0 max hyp exceed
    W = np.copy(W_prev)
    for i in range(0, max_hyp-1):
        W = -1 + 2 * np.random.random((data_x.shape[0], k))
        points = data_x[:, np.random.choice(np.arange(0, W.shape[1]), size=k, replace=True)]
        W[-1, :] = np.sum(W[0:-1, :] * points[0:-1, :], axis=0)
        W = learn_hyperplanes(W, data_x, data_y, 0.001, pdf, visualise)
        bic_new = bic(W, data_x, data_y)
        if bic_new > bic_prev:
            ter_code = 1
            W = W_prev
            break
        k=k+1
        W_prev = W
        bic_prev = bic_new
    if visualise == 1:
        pdf.output('spla_output' + '.pdf')
    return W, ter_code

def train_1(data_x, data_y, max_hyp, visualise = False):
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    W_prev = logistic_regression(data_x, data_y, pdf)
    _, _, bic_prev = bic(W_prev, data_x, data_y)
    k=2
    ter_code=0 #1 bic termination, 0 max hyp exceed
    W = np.copy(W_prev)
    for i in range(0, max_hyp-1):
        W = logistic_regression_k(data_x, data_y, k, pdf)
        W = learn_hyperplanes(W, data_x, data_y, 0.001, pdf, visualise)
        _, _, bic_new = bic(W, data_x, data_y)
        if bic_new > bic_prev:
            ter_code = 1
            W = W_prev
            break
        k=k+1
        W_prev = W
        bic_prev = bic_new
    if visualise == 1:
        pdf.output('spla_regression_fix' + '.pdf')
    return W, ter_code

def train_spla_2(data_x, data_y, max_hyp, visualise = False):
    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)
    W_prev = logistic_regression(data_x, data_y, pdf)
    _, _, bic_prev = bic(W_prev, data_x, data_y)
    k=2
    ter_code=0 #1 bic termination, 0 max hyp exceed
    W = np.copy(W_prev)
    for i in range(0, max_hyp-1):
        W = logistic_regression_k(data_x, data_y, k, pdf)
        W = learn_hyperplanes_hess(W, data_x, data_y, 0.001, pdf, visualise)
        _, _, bic_new = bic(W, data_x, data_y)
        if bic_new > bic_prev:
            ter_code = 1
            W = W_prev
            break
        k=k+1
        W_prev = W
        bic_prev = bic_new
    if visualise == 1:
        pdf.output('spla_2_circles' + '.pdf')
    return W, ter_code