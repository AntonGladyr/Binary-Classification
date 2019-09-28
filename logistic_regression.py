import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate_function, condition, itr=True):
        self.lr_func = learning_rate_function
        self.parameters = None
        self.condition = condition
        self.itr = itr

    def __logistic_function(self, training_example):
        a = np.dot(training_example, self.parameters.T)
        return 1.0/(1.0 + np.exp(-a))

    def __update_step(self, features, targets, k):
        total_sum = np.zeros((features.shape[1]))
        for row in range(features.shape[0]):
            total_sum = total_sum + features[row, :] * (targets[row] - self.__logistic_function(features[row, :]))
        return self.parameters + self.lr_func(k) * total_sum

    def __fit_itr(self, features, targets, maxitr):
        self.parameters = np.zeros((features.shape[1]))
        for k in range(maxitr):
            self.parameters = self.__update_step(features, targets, k)

    def __fit_threshold(self, features, targets, atol):
        self.parameters = np.zeros((features.shape[1]))
        prev_parameters = np.ones((features.shape[1]))
        k = 0
        while False in np.isclose(prev_parameters, self.parameters, atol=atol):
            prev_parameters = self.parameters
            self.parameters = self.__update_step(features, targets, k)
            k += 1

    def fit(self, features, targets):
        if(self.itr):
            self.__fit_itr(features, targets, self.condition)
        else:
            self.__fit_threshold(features, targets, self.condition)

    def predict(self, features):
        predictions_list = []
        for row in range(features.shape[0]):
            predictions_list.append(np.round(self.__logistic_function(features[row, :])))
        return np.asarray(predictions_list)
