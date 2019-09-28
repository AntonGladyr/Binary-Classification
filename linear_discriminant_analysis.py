"""
Linear Discriminant Analysis
"""

import numpy as np


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
        _, columns_amount = X.shape
        covariance_matrix = np.zeros((columns_amount, columns_amount))
        for cl, mv in zip(range(self.classes_amount), mean_vectors):
            class_sc_mat = np.zeros((columns_amount, columns_amount))  # scatter matrix for every class
            for row in X[y == cl]:
                row, mv = row.reshape(columns_amount, 1), mv.reshape(columns_amount, 1)  # make column vectors
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
        for MU in range(self.classes_amount):
            SIGMA_inv = np.linalg.inv(self.covariance_matrix)
            m, _ = self.covariance_matrix.shape
            denominator = np.sqrt((2 * np.pi)**m) * np.sqrt(np.linalg.det(self.covariance_matrix))
            exponent = -(1 / 2) * ((X - MU).T @ SIGMA_inv @ (X - MU))
            return float((1. / denominator) * np.exp(exponent))


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

    def linear_discriminant_func(self, X, MU, proba):
        #linear discriminant score function for a given class "k"
        sigma_inv = np.linalg.inv(self.covariance_matrix)
        return (X.T @ sigma_inv @ MU - 1 / 2 * MU.T @ sigma_inv @ MU + np.log(proba)).flatten()[0]


    def fit(self, X, y):
        self.classes_amount = np.unique(y).size
        self.mean_vectors = self.__compute_mean_vectors(X, y)
        self.covariance_matrix = self.__compute_covariance_matrix(X, y, self.mean_vectors)
        self.likelihood_list = self.__compute_likelihood_list(y)


    def predict(self, X):
        # 1 if decision_boundary > 0 else 0
        log_ratios = np.array([self.compute_decision_boundary(xx) for xx in X]).flatten()
        log_ratios[log_ratios > 0] = 1
        log_ratios[log_ratios < 0] = 0
        return log_ratios

    def predict_multiple_lda(self, X):
        # Returns the class for which the the linear discriminant score function is largest
        scores_list = []

        for mu, proba in zip(self.mean_vectors, self.likelihood_list):
            score = self.linear_discriminant_func(X, mu, proba)
            scores_list.append(score)
        return np.argmax(scores_list)
