#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:mathilda 
@Email: 119020045@link.cuhk.edu.com
@file: eigengapRatio.py
@time: 2021/08/27

Notes: objective function: log(tr((v_k+1)'L(v_k+1)) - tr((v_k)'L(v_k))) - log(tr((v_k+1)'L(v_k+1)))
        gradient:= 1/[tr() - tr()] * [(v_k+1)(v_k+1)' - (v_k)(v_k)'] * gK_L * gGamma_K
                - 1/tr() * (v_k+1)(v_k+1)' * gK_L * gGamma_K

        np.exp(-gamma * d(X,X) ** 2)
"""

from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat
from scipy import linalg
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import pandas as pd
from sklearn import metrics
# import keras
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
# from scipy.optimize import line_search
# from tqdm import tqdm


def COIL20(num_samples = -1, seed = 0):
    data = loadmat('/Users/mathilda/OneDrive - CUHK-Shenzhen/Research/Github_updates/laplacian_t-SNE/code/data/COIL20.mat')
    if num_samples > 0:
        np.random.seed(seed)
        select = np.random.choice(np.arange(len(data)), size = num_samples)
        X = data['fea'][select]
        y = data['gnd'][select]
    else:
        X, y = data['fea'],data['gnd']

    if num_samples < 0: num_samples = X.shape[0]
    print(f'--------Finish loading COIL20 dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y.ravel()

def MNIST(num_samples = -1, seed = 0):
    mnist_image_file = '/Users/mathilda/OneDrive - CUHK-Shenzhen/Research/Github_updates/laplacian_t-SNE/code/data/MNIST_train'
    mnist_label_file = '/Users/mathilda/OneDrive - CUHK-Shenzhen/Research/Github_updates/laplacian_t-SNE/code/data/MNIST_train_label'
    NUM_SAMPLE = 2000
    X, y = list(), list()
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()

    image_file = image_file[16:]
    label_file = label_file[8:]

    if num_samples <= 0: num_samples = len(label_file)
    for i in range(num_samples):
        label = int(label_file[i])
        image_list = [int(item) for item in image_file[i * 784:i * 784 + 784]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(28 * 28)
        X.append(image_np)
        y.append(label)

    X, y = np.array(X), np.array(y)
    print(f'--------Finish loading MNIST dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for (i, j) in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def power_diag(D,power):
    D_new = np.diag(np.power(np.diag(D),power))
    return D_new

def gk_L(A, D, coef):
    n = A.shape[0]
    U0 = -0.5 * power_diag(D,-1.5) @ A @ power_diag(D,-0.5) * coef
    U1 = -0.5 * power_diag(D,-0.5) @ A @ power_diag(D,-1.5) * coef
    U0 = np.tile(U0.sum(axis=1), (n, 1))
    U1 = np.tile(U1.sum(axis=0), (n, 1)).T
    grad_LK = -(U0 + U1 + power_diag(D,-1) * coef)
    return grad_LK

def gGamma_K(K, Dist):
    grad = - K * Dist
    return grad

def line_search(obj_F, grad_F, x, d):
    sigma = 0.8; alpha = 1e-3; gamma = 1e-4
    count = 1
    while True:
        if (obj_F(x + alpha * d) - obj_F(x)) >= (gamma * alpha * grad_F(x).T @ d): break
        print(f'Step size {alpha} fails, current gap is {obj_F(x + alpha * d) - obj_F(x)}')
        alpha *= sigma
        count += 1
        if count > 40: break

    print(obj_F(x + alpha * d))
    print(obj_F(x))
    print(f'Step size: {alpha}')
    return alpha

def eigenGapRatio(X, k, gamma):
    n = X.shape[0]
    # Dist = pairwise_distances(X, squared=True) #D
    Dist = squareform(pdist(X, "sqeuclidean"))

    def obj_grad(x = gamma):
        K = np.exp(- x * Dist)  # K_j|i = P_j|i
        A = (K + K.T) * 0.5 - np.eye(n)
        D = np.diag(A.sum(axis=1))
        L = np.eye(n) - power_diag(D, -0.5) @ A @ power_diag(D, -0.5)
        lam, eigenVectors = linalg.eigh(L, subset_by_index=[k-1, k])
        eigenGap = lam[-1] - lam[-2]; lam_Kplus1 = lam[-1]
        eigenGapRatio = eigenGap / lam_Kplus1
        print('lam_Kplus1 is', lam_Kplus1)
        print('Current eigenGapRatio is ', eigenGapRatio)
        eig_V = eigenVectors[:, -2]; eig_V_kp1 = eigenVectors[:, -1]
        inv_trace1 = eigenGap ** (-1) * (eig_V_kp1 @ eig_V_kp1.T - eig_V @ eig_V.T)
        inv_trace2 = lam_Kplus1 ** (-1) * (eig_V_kp1 @ eig_V_kp1.T)

        grad = gk_L(A, D, (inv_trace1 - inv_trace2)) * gGamma_K(K, Dist) @ np.ones((n, 1))

        return grad.flatten()

    def obj_func(x = gamma):
        K = np.exp(- x * Dist)  # K_j|i = P_j|i
        A = (K + K.T) * 0.5 - np.eye(n)
        D = np.diag(A.sum(axis=1))
        L = np.eye(n) - power_diag(D, -0.5) @ A @ power_diag(D, -0.5)
        lam, eig_V = linalg.eigh(L, subset_by_index=[k-1, k])
        eigenGap = lam[-1] - lam[-2]
        lam_Kplus1 = lam[-1]
        eigenGapRatio = eigenGap / lam_Kplus1

        return np.log(eigenGapRatio).flatten()

    K = np.exp(- gamma * Dist)  # K_j|i = P_j|i
    A = (K + K.T) * 0.5 - np.eye(n)
    D = np.diag(A.sum(axis=1))
    L = np.eye(n) - power_diag(D, -0.5) @ A @ power_diag(D, -0.5)
    lam, eig_V = linalg.eigh(L, subset_by_index=[0, k])

    grad = obj_grad(gamma)
    # stepsize = line_search(obj_func, obj_grad, gamma, grad)
    stepsize = 1e-5
    # stepsize = np.linalg.norm(grad) * 1e-7
    full_grad = stepsize * grad
    eigenGapRatio = np.exp(obj_func(gamma))

    return eigenGapRatio, full_grad, lam


def main(score_dict, num_iter = 20):
    X, y = COIL20(num_samples=-1, seed=0)
    # X, y = MNIST(num_samples=1000, seed=0)
    print(f'---- Shape of X: {X.shape} -----')
    n = X.shape[0]
    k = 20
    # learning_rate = 1e-5
    # gamma = np.ones(n)* 1e-1
    gamma = np.ones(n) * 1e-1
    gamma_list, ratio_list, lam_list= [], [], []
    update = np.zeros(n)
    it = 0
    for i in range(num_iter):
        if i < 15:
            momentum = 0.5
        else:
            momentum = 0.8
        print(f'----- At No.{i+1} iteration -----')
        ratio, grad, lam = eigenGapRatio(X, k, gamma)
        print(grad[:2])
        update = momentum * update + grad # maximize
        gamma += update
        plt.hist(gamma, bins=20)
        plt.title(f'No. {i+1} iteration')
        plt.show()

        gamma_list.append(gamma)
        ratio_list.append(ratio)
        lam_list.append(lam)
        it += 1

        clustering = SpectralClustering(n_clusters=k, random_state = 0, gamma = gamma).fit_predict(X)

        labels_true,  labels = y, clustering
        # score_dict['Homogeneity'].append(metrics.homogeneity_score(labels_true, labels))
        # score_dict['Completeness'].append(metrics.completeness_score(labels_true, labels))
        # score_dict['V-measure'].append(metrics.v_measure_score(labels_true, labels))
        # score_dict['Adjusted Rand Index'].append(metrics.adjusted_rand_score(labels_true, labels))
        score_dict['Accuracy'].append(cluster_acc(labels_true, labels))
        score_dict['Adjusted Mutual Information'].append(metrics.adjusted_mutual_info_score(labels_true, labels))
        # score_dict['Silhouette Coefficient'].append(metrics.silhouette_score(X, labels))


    # finally:
    plt.plot(range(num_iter), ratio_list)
    plt.title(f'Eigengap Ratio')
    plt.xlabel = 'Num of Iterations'
    plt.show()

    return score_dict, lam_list


if __name__ == '__main__':
    score_dict = {#'Homogeneity':[],
                  # 'Completeness':[],
                  # 'V-measure':[],
                  # 'Adjusted Rand Index': [],
                  'Adjusted Mutual Information': [],
                  # 'Silhouette Coefficient': [],
                  'Accuracy': []
    }
    num_iter = 20
    score_dict, lam_list = main(score_dict, num_iter)
    plt.plot(range(num_iter), pd.DataFrame(score_dict))
    plt.title(f'Clustering Metrics (COIL20, n_clusters = {num_iter})')
    plt.legend(list(score_dict.keys()))
    plt.xlabel = 'Num of Iterations'
    plt.show()
    lam_df = pd.DataFrame(lam_list)
    lam_df.plot()
    plt.show()

