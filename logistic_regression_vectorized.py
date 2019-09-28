import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate_function, condition, itr=True):
        self.lr_func = learning_rate_function
        self.parameters = None
        self.condition = condition
        self.itr = itr

    def __logistic_function(self, features):
        a = np.dot(features, self.parameters)
        return 1.0/(1 + np.exp(-a))

    def __update_step(self, features, targets, k):
        # step 1: sigmoid of all weights(dot)features representing probabilities dim(n, 1)
        probabilities = self.__logistic_function(features)

        # step 2: gradient. Sum[x_i( y_i - sigmoid(w*x_i) ), {i, 1, n}] =: X(dot)probabilities
        # dim(m, 1)
        gradient = np.dot(features.T, targets - probabilities)

        # step 3: multiply by learning rate. dim(m, 1)
        gradient *= self.lr_func(k)

        # step 4: update weights
        return self.parameters + gradient

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
        return np.round(self.__logistic_function(features))


if __name__ == "__main__":
    import sys
    from cleaner import cleanWineDataset
    from cleaner import cleanCancerDataset
    from runner import split_dataset, evaluate_acc
    from learning_rates import learning_rate_func1

    dataset = None
    if sys.argv[1] == 'wine_dataset':
        dataset = cleanWineDataset()
        TARGET_INDEX = 11
    elif sys.argv[1] == 'cancer_dataset':
        dataset = cleanCancerDataset()
        TARGET_INDEX = 9
    else:
        sys.exit(1)

    features = dataset[:, :TARGET_INDEX]
    targets = dataset[:, TARGET_INDEX]
    X_train, X_val, Y_train, Y_val = split_dataset(features, targets, 0.9)

    model = LogisticRegression(learning_rate_func1, 100000, itr=True)
    model.fit(X_train, Y_train)
    print(model.parameters)
    print('maxitr = %d, accuracy = %f' % (100000, evaluate_acc(model.predict(X_val), Y_val)))

    sys.exit(0)
