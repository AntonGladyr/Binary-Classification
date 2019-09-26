"""
Linear Discriminant Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.mean_vectors = np.array([])
        self.covariance_matrix = np.array([])
        self.likelihood_list = np.array([])
        self.classes_amount = 0

    def __compute_mean_vectors(self, X, y):
        mean_vectors = []
        for cl in range(self.classes_amount):
            mean_vectors.append(np.mean(X[y == cl], axis=0))
        return np.array(mean_vectors)

    def __compute_covariance_matrix(self, X, y, mean_vectors):
        covariance_matrix = np.zeros((9, 9))
        for cl, mv in zip(range(self.classes_amount), mean_vectors):
            class_sc_mat = np.zeros((9, 9))  # scatter matrix for every class
            for row in X[y == cl]:
                row, mv = row.reshape(9, 1), mv.reshape(9, 1)  # make column vectors
                class_sc_mat += (row - mv).dot((row - mv).T)
            covariance_matrix += class_sc_mat  # sum class scatter matrices
        covariance_matrix = (1 / (X[:, 0].size - 2)) * covariance_matrix
        # alternative for the cycle
        #covariance_matrix = np.cov(X.T, bias=True, ddof=2)
        return covariance_matrix

    def __compute_likelihood_list(self, y):
        likelihood_list = []
        for cl in range(self.classes_amount):
            likelihood_list.append(y[y == cl].size / y.size)
        return likelihood_list


    def gaussian_pdf(self, X):
        cost_list = np.array([])
        for MU in range(self.classes_amount):
            SIGMA_inv = np.linalg.inv(self.covariance_matrix)
            m, _ = self.covariance_matrix.shape
            denominator = np.sqrt((2 * np.pi)**m) * np.sqrt(np.linalg.det(self.covariance_matrix))
            exponent = -(1 / 2) * ((X - MU).T @ SIGMA_inv @ (X - MU))
            cost = float((1. / denominator) * np.exp(exponent))
            cost_list = np.append(cost_list, cost)
        return 1 if cost_list[0] > cost_list[1] else 0

    def compute_decision_boundary(self, X):
        # TODO: optimize
        P0 = self.likelihood_list[0]
        P1 = self.likelihood_list[1]
        MU0 = self.mean_vectors[0].reshape(-1, 1)
        MU1 = self.mean_vectors[1].reshape(-1, 1)
        sigma_inv = np.linalg.inv(self.covariance_matrix)
        W = sigma_inv @ (MU1 - MU0)
        log_ratio = ((np.log(P1 / P0) - 1 / 2 * MU1.T @ sigma_inv @ MU1 + 1 / 2 * \
                     MU0.T @ sigma_inv @ MU0) + X.T @ W).flatten()
        #log_ratio = (np.log(P1 / P0) - 1/2 * (MU1 + MU0).T @ sigma_inv@(MU1 - MU0) + X.T @ sigma_inv@  (MU1 - MU0))
        return log_ratio

    def copmute_multiple_decision_boundary(self, X):
        return
        #linear discriminant score function for a given class "k"

        #return (np.log(pi_k) - 1 / 2 * (MU_k).T @ np.linalg.inv(SIGMA) @ (MU_k) + X.T @ np.linalg.inv(SIGMA) @ (
        #    MU_k)).flatten()[0]



    def fit(self, X, y):
        self.classes_amount = np.unique(y).size
        self.mean_vectors = self.__compute_mean_vectors(X, y)
        self.covariance_matrix = self.__compute_covariance_matrix(X, y, self.mean_vectors)
        self.likelihood_list = self.__compute_likelihood_list(y)

    def predict_by_gaussian(self, X):
        predictions = np.array([self.gaussian_pdf(xx) for xx in X])
        return predictions

    def predict(self, X):
        # 1 if decision_boundary > 0 else 0
        log_ratios = np.array([self.compute_decision_boundary(xx) for xx in X]).flatten()
        log_ratios[log_ratios > 0] = 1
        log_ratios[log_ratios < 0] = 0
        return log_ratios

    def predict_multiple_lda(self, X):
        scores_list = np.array([])


    def evaluate_acc(self, y, predictions):
        return np.mean(predictions == y)
