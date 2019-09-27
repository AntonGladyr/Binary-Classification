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


    #visualizer.visualize_predictions(X, y, lda)
    #visualizer.visualize(wine_dataset, WINE_COLUMNS)
    #visualizer.visualize(cancer_dataset, CANCER_COLUMNS)
    #visualizer.visualize_univariate(wine_dataset, WINE_COLUMNS)



if __name__ == '__main__':
    main()
