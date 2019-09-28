#!/usr/bin/env python
"""
COMP 551
MiniProject 1

https://www.cs.mcgill.ca/~wlh/comp551/files/miniproject1_spec.pdf

This mini project implements two linear classification techniques — logistic regression and linear discriminant
analysis (LDA) as well as runs these two algorithms on two distinct datasets.

Authors:
            Anton Gladyr    anton.gladyr@mail.mcgill.ca
            Saleh Bakhit    saleh.bakhit@mail.mcgill.ca
            Vasu Khanna     vasu.khanna@mail.mcgill.ca

Created on 16 Sep 2019
"""

import numpy as np
import cleaner
import time
import visualizer
from linear_discriminant_analysis import LinearDiscriminantAnalysis

CANCER_COLUMNS = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size',
               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']
WINE_COLUMNS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
CANCER_CLASS_INDEX = 9


def k_fold_runner(model, dataset, k, target_index):
    np.random.shuffle(dataset)

    partition_size = dataset.shape[0] // k
    starting_index = 0
    for i in range(k):
        dataset_val = dataset[starting_index:partition_size * (i + 1), :]
        dataset_train = np.delete(dataset, np.s_[starting_index:partition_size * (i + 1)], axis=0)

        features_train = dataset_train[:, :target_index]
        targets_train = dataset_train[:, target_index]

        features_val = dataset_val[:, :target_index]
        targets_val = dataset_val[:, target_index]

        model.fit(features_train, targets_train)
        print('itr %d with accuracy %f' % (i, np.mean(model.predict(features_val) == targets_val)))

        starting_index = partition_size * (i + 1)

def main():
    wine_dataset = cleaner.cleanWineDataset()
    cancer_dataset = cleaner.cleanCancerDataset()
    X_wine = wine_dataset[:, :-1]
    y_wine = np.array(wine_dataset[:, -1])
    X_cancer = cancer_dataset[:, :-1]
    y_cancer = np.array(cancer_dataset[:, -1])
    lda = LinearDiscriminantAnalysis()

    # test LDA on wine dataset
    start = time.time()
    lda.fit(X_wine, y_wine)
    end = time.time()
    print('\nLinear discriminant analysis model ~wine dataset~:')
    print('\tComputation time: {0}'.format(end - start))
    predictions = lda.predict(X_wine)
    accuracy = lda.evaluate_acc(y_wine, predictions)
    print('\tAccuracy of predictions on the wine dataset: {0}\n'.format(accuracy))

    # test LDA on cancer dataset
    start = time.time()
    lda.fit(X_cancer, y_cancer)
    end = time.time()
    print('\nLinear discriminant analysis model ~cancer dataset~:')
    print('\tComputation time: {0}'.format(end - start))
    predictions = lda.predict(X_cancer)
    accuracy = lda.evaluate_acc(y_cancer, predictions)
    print('\tAccuracy of predictions on the cancer dataset: {0}'.format(accuracy))
    predictions = [lda.predict_multiple_lda(xx) for xx in X_cancer]
    accuracy = lda.evaluate_acc(y_cancer, predictions)
    print('\tAccuracy of predictions on the cancer dataset ~multiple classes implementation~: {0}\n'.format(accuracy))

    #changing the values of wine dataset for improving the accuracy
    X_wine[X_wine[:, 6] > 100, 6] = 100
    lda.fit(X_wine, y_wine)
    predictions = lda.predict(X_wine)
    accuracy = lda.evaluate_acc(y_wine, predictions)
    print('\tAccuracy of predictions on the wine dataset: {0}'.format(accuracy))

    #run k-fold cross validation
    k_fold_runner(lda, cancer_dataset, 5, CANCER_CLASS_INDEX)


    #visualizer.visualize_predictions(X, y, lda)
    #visualizer.visualize(wine_dataset, WINE_COLUMNS)
    #visualizer.visualize(cancer_dataset, CANCER_COLUMNS)
    #visualizer.visualize_univariate(wine_dataset, WINE_COLUMNS)



if __name__ == '__main__':
    main()
