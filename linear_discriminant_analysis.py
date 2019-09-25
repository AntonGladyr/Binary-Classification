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

    def __compute_mean_vectors(self, X, y):
        mean_vectors = []
        classes = np.unique(y).size
        for cl in range(classes):
            mean_vectors.append(np.mean(X[y == cl], axis=0))
        return np.array(mean_vectors)

    def __compute_covariance_matrix(self, X, y, mean_vectors):
        # TODO refactor
        covariance_matrix = np.zeros((9, 9))
        for cl, mv in zip(range(2), mean_vectors):
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
        classes = np.unique(y).size
        for cl in range(classes):
            likelihood_list.append(y[y == cl].size / y.size)
        return likelihood_list


    def gaussian_pdf(self, X):
        cost_list = np.array([])
        for MU in range(self.mean_vectors.size):
            SIGMA_inv = np.linalg.inv(self.covariance_matrix)
            m, _ = self.covariance_matrix.shape
            denominator = np.sqrt((2 * np.pi)**m) * np.sqrt(np.linalg.det(self.covariance_matrix))
            exponent = -(1 / 2) * ((X - MU).T @ SIGMA_inv @ (X - MU))
            cost = float((1. / denominator) * np.exp(exponent))
            cost_list = np.append(cost_list, cost)
        return cost_list

    def compute_decision_boundary(self, X):
        # TODO: optimize
        P0 = self.likelihood_list[0]
        P1 = self.likelihood_list[1]
        MU0 = self.mean_vectors[0].reshape(-1, 1)
        MU1 = self.mean_vectors[1].reshape(-1, 1)
        sigma_inv = np.linalg.inv(self.covariance_matrix)
        W = sigma_inv @ (MU1 - MU0)
        log_ratio = ((np.log(P1 / P0) - 1 / 2 * MU1.T @ sigma_inv @ MU1 + 1 / 2 * \
                     MU0.T @ sigma_inv @ MU0) + X @ W).flatten()
        #log_ratio = (np.log(P1 / P0) - 1/2 * (MU1 + MU0).T @ sigma_inv@(MU1 - MU0) + X.T @ sigma_inv@  (MU1 - MU0))
        return log_ratio

    def fit(self, X, y):
        self.mean_vectors = self.__compute_mean_vectors(X, y)
        self.covariance_matrix = self.__compute_covariance_matrix(X, y, self.mean_vectors)
        self.likelihood_list = self.__compute_likelihood_list(y)

    def predict(self, X):
        # 1 if decision_boundary > 0 else 0
        log_ratios = self.compute_decision_boundary(X)
        log_ratios[log_ratios > 0] = 1
        log_ratios[log_ratios < 0] = 0
        return log_ratios

    def evaluate_acc(self, y, predictions):
        return np.mean(predictions == y)

    def visualize_on_plot(self, X, y):
        # TODO: fix the visualization
        #N = 100
        #X = np.linspace(3, 8, N)
        #Y = np.linspace(1.5, 5, N)
        #X, Y = np.meshgrid(X, Y)

        # input values
        z = self.compute_decision_boundary(X.T)
        print(z.shape)
        return
        cm_bright = colors.ListedColormap(['#FF0000', '#0000FF'])  # just colors
        plt.contourf(xx, yy, B, alpha=0.4, cmap=cm_bright)  # plot the contours using our classifier results
        #plt.scatter(x0, x1, c=y, s=20, cmap=cm_bright)  # scatter plot of true data

        # labelling the plot
        plt.xlabel("benign")
        plt.ylabel("malignant")
        plt.title("Classifier boundries vs true data results")
        plt.show()
