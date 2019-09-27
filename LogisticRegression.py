import sys
import numpy as np
from cleaner import cleanWineDataset
from cleaner import cleanCancerDataset
from sklearn.linear_model import LogisticRegression as logReg
import time

class LogisticRegression():
    def __init__(self, learning_rate_function):
        self.lr_func = learning_rate_function

        self.parameters = None
        self.predictions = None

    def _logistic_function(self, training_example):
        a = np.dot(training_example, self.parameters.T)
        return 1.0/(1.0 + np.exp(-a))

    def _update_step(self, features, targets, k):
        total_sum = np.zeros((features.shape[1]))
        for row in range(features.shape[0]):
            total_sum = total_sum + features[row, :] * (targets[row] - self._logistic_function(features[row, :]))
        return self.parameters + self.lr_func(k) * total_sum

    def fit(self, features, targets, maxit):
        self.parameters = np.zeros((features.shape[1]))
        for k in range(maxit):
            self.parameters = self._update_step(features, targets, k)

    def fit(self, features, targets, atol):
        self.parameters = np.zeros((features.shape[1]))
        prev_parameters = np.ones((features.shape[1]))
        k = 0
        while False in np.isclose(prev_parameters, self.parameters, atol=atol):
            prev_parameters = self.parameters
            self.parameters = self._update_step(features, targets, k)
            k += 1

    def predict(self, features):
        predictions_list = []
        for row in range(features.shape[0]):
            predictions_list.append(np.round(self._logistic_function(features[row, :])))
        return np.asarray(predictions_list)


def split_dataset(features, targets, pct):
    train_pct_index = int(pct * len(features))
    X_train, X_val = features[:train_pct_index, :], features[train_pct_index:, :]
    Y_train, Y_val = targets[:train_pct_index], targets[train_pct_index:]
    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__":
    from Training import learning_rate_constant, maxit, learning_rate2

    if len(sys.argv) < 2:
        print('no dataset specified')
        sys.exit(1)

    dataset = None
    TARGET_INDEX = None
    if sys.argv[1] == "wine_dataset":
        dataset = cleanWineDataset()
        TARGET_INDEX = 11
    elif sys.argv[1] == "cancer_dataset":
        dataset = cleanCancerDataset()
        TARGET_INDEX = 'TBD'
    else:
        print('dataset specified not one of (wine_dataset, cancer_dataset)')
        sys.exit(1)

    features = dataset[:, :TARGET_INDEX]
    targets = dataset[:, TARGET_INDEX]
    X_train, X_val, Y_train, Y_val = split_dataset(features, targets, 0.9)

    logisticRegression = LogisticRegression(learning_rate_constant)
    
    start_time = time.time()
    logisticRegression.fit(X_train, Y_train, 1e-3)
    print("--- 1e-3 took %s seconds ---" % (time.time() - start_time))
    a = logisticRegression.parameters

    start_time = time.time()
    logisticRegression.fit(X_train, Y_train, 1e-4)
    print("--- 1e-4 took %s seconds ---" % (time.time() - start_time))
    b = logisticRegression.parameters

    start_time = time.time()
    logisticRegression.fit(X_train, Y_train, 1e-5)
    print("--- 1e-5 took %s seconds ---" % (time.time() - start_time))
    c = logisticRegression.parameters

    start_time = time.time()
    logisticRegression.fit(X_train, Y_train, 1e-6)
    print("--- 1e-6 took %s seconds ---" % (time.time() - start_time))
    d = logisticRegression.parameters

    # start_time = time.time()
    # logisticRegression.fit(X_train, Y_train)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # e = logisticRegression.parameters

    import pdb; pdb.set_trace()

    sys.exit(0)
