#!/usr/bin/env python
import numpy as np
from cleaner import cleanWineDataset
from cleaner import cleanCancerDataset


class LogisticRegression:
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def predict(self, features, weights):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        z = np.dot(features, weights)
        return self.sigmoid(z)

    def cost_function(self, features, labels, weights):
        '''
        Using Mean Absolute Error

        Features:(100,3)
        Labels: (100,1)
        Weights:(3,1)
        Returns 1D matrix of predictions
        Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
        '''
        observations = len(labels)

        predictions = self.predict(features, weights)

        # Take the error when label=1
        class1_cost = -labels * np.log(predictions)

        # Take the error when label=0
        class2_cost = (1 - labels) * np.log(1 - predictions)

        # Take the sum of both costs
        cost = class1_cost + class2_cost

        # Take the average cost
        cost = cost.sum() / observations

        return cost

    def update_weights(self, features, labels, weights, lr):
        '''
        Vectorized Gradient Descent

        Features:(200, 3)
        Labels: (200, 1)
        Weights:(3, 1)
        '''
        N = len(features)

        # 1 - Get Predictions
        predictions = self.predict(features, weights)

        # 2 Transpose features from (200, 3) to (3, 200)
        # So we can multiply w the (200,1)  cost matrix.
        # Returns a (3,1) matrix holding 3 partial derivatives --
        # one for each feature -- representing the aggregate
        # slope of the cost function across all observations
        gradient = np.dot(features.T, predictions - labels)

        # 3 Take the average cost derivative for each feature
        gradient /= N

        # 4 - Multiply the gradient by our learning rate
        gradient *= lr

        # 5 - Subtract from our weights to minimize cost
        weights -= gradient

        return weights

    def decision_boundary(self, prob):
        return 1 if prob >= .5 else 0

    def classify(self, predictions):
        '''
        input  - N element array of predictions between 0 and 1
        output - N element array of 0s (False) and 1s (True)
        '''
        decision_boundary = np.vectorize(self.decision_boundary)
        return decision_boundary(predictions).flatten()

    def train(self, features, labels, weights, lr, iters):
        cost_history = []

        for i in range(iters):
            weights = self.update_weights(features, labels, weights, lr)

            # Calculate error for auditing purposes
            cost = self.cost_function(features, labels, weights)
            cost_history.append(cost)

            # Log Progress
            if i % 1000 == 0:
                print('iter: ' + str(i) + " cost: " + str(cost))

        return weights, cost_history

    def accuracy(self, predicted_labels, actual_labels):
        diff = predicted_labels - actual_labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def main():
    dataset = cleanWineDataset()
    lr = LogisticRegression()
    features = dataset[:, :11]
    targets = dataset[:, 11]
    beta = np.zeros((features.shape[1]))
    weights, cost_history = lr.train(features, targets, beta, 0.00001, 50000)
    print('accuracy: {0}'.format(np.max(lr.predict(features, weights))))

if __name__ == '__main__':
     main()