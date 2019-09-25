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
import time
from pandas import read_csv as read
from linear_discriminant_analysis import LinearDiscriminantAnalysis

WINE_DATA_PATH = "./resources/winequality-red.csv"
CANCER_DATA_PATH = "./resources/breast-cancer-wisconsin.data"
WINE_QUALITY_INDEX = 11
CANCER_CLASS_INDEX = 10
label_dict = {'benign': 2, 'malignant': 4}


def clean_data(wine_data, cancer_data):
    # TODO refactor this function
    # checking for missing values in the wine dataset
    # if the resulting array is not empty
    if np.argwhere(np.isnan(wine_data)).size > 0:
        # deleting rows with missing values
        #   np.isnan  - returns boolean with True where NaN, and False elsewhere;
        #   .any(axis=1)  - reduces an m*n array to n with an logical or operation on the whole rows;
        #   ~  - inverts True/False
        wine_data = np.array(wine_data[~np.isnan(wine_data).any(axis=1)])

    # converting quality ratings of wines to binary values
    wine_data[:, WINE_QUALITY_INDEX][wine_data[:, WINE_QUALITY_INDEX] <= 5] = 0
    wine_data[:, WINE_QUALITY_INDEX][wine_data[:, WINE_QUALITY_INDEX] >= 6] = 1

    # checking for missing/malformed values in the cancer dataset
    if np.argwhere(cancer_data == '?').size > 0:
        cancer_data = np.array(cancer_data[~(cancer_data == '?').any(axis=1)])
    # converting string-values to int type
    cancer_data = cancer_data.astype(int)
    # converting cancer classes to binary values
    cancer_data[:, CANCER_CLASS_INDEX][cancer_data[:, CANCER_CLASS_INDEX] == label_dict['benign']] = 0
    cancer_data[:, CANCER_CLASS_INDEX][cancer_data[:, CANCER_CLASS_INDEX] == label_dict['malignant']] = 1
    # removing sample code numbers
    cancer_data = cancer_data[:, 1:]
    return wine_data, cancer_data


def main():
    wine_dataset = np.array(read(WINE_DATA_PATH, delimiter=";"))
    cancer_dataset = np.array(read(CANCER_DATA_PATH, delimiter=",", header=None))
    wine_dataset, cancer_dataset = clean_data(wine_dataset, cancer_dataset)

    X = cancer_dataset[:, :-1]
    y = np.array(cancer_dataset[:, -1])
    start = time.time()
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    end = time.time()
    print('\nComputation time: {0}'.format(end - start))
    predictions = lda.predict(X)
    accuracy = lda.evaluate_acc(y, predictions)
    print('Accuracy for the linear discriminant analysis model: {0}\n'.format(accuracy))


if __name__ == '__main__':
    main()
