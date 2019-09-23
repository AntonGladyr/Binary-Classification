import sys
import numpy as np
from cleaner import cleanWineDataset
from cleaner import cleanCancerDataset


class LogisticRegression():
    def __init__(self, num_iterations, learning_rate_function):
        self.num_iterations = num_iterations
        self.lr_func = learning_rate_function

        self.parameters = None

    def __logistic_function(self, training_example):
        a = np.dot(self.parameters, training_example)
        print(self.parameters.T)
        print(training_example)
        print('\n')
        return 1/(1 + np.exp(-a))

    def __update_step(self, features, targets, k):
        import pdb; pdb.set_trace()
        total_sum = 0
        for row in range(1500):
            # print(self.__logistic_function(features[row]))
            total_sum += features[row] * (targets[row] - self.__logistic_function(features[row]))
        # import pdb; pdb.set_trace()
        self.parameters = self.parameters + self.lr_func(k) * total_sum

    def fit(self, features, targets):
        self.parameters = np.zeros((features.shape[1], 1))
        for k in range(self.num_iterations):
            self.__update_step(features, targets, k)

    def predict(self, features):
        # TODO: return np.array of predicted y to each row in features to predict
        return self.__logistic_function(features)


if __name__ == "__main__":
    from Training import learning_rate_constant, NUM_ITERATIONS

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

    logisticRegression = LogisticRegression(NUM_ITERATIONS, learning_rate_constant)
    logisticRegression.fit(features, targets)
    print(logisticRegression.parameters)

    sys.exit(0)
