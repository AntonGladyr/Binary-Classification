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
from pandas import read_csv as read

WINE_DATA_PATH = "./resources/winequality-red.csv"
CANCER_DATA_PATH = "./resources/breast-cancer-wisconsin.data"
QUALITY_INDEX = 11


def process_data():
    # TODO: move data preprocessing here
    print()


def main():
    wine_dataset = np.array(read(WINE_DATA_PATH, delimiter=";"))
    cancer_dataset = np.array(read(CANCER_DATA_PATH, delimiter=",", header=None))

    # checking for missing values in the wine dataset
    # if the resulting array is not empty
    if np.argwhere(np.isnan(wine_dataset)).size > 0:
        # deleting rows with missing values
        #   np.isnan  - returns boolean with True where NaN, and False elsewhere;
        #   .any(axis=1)  - reduces an m*n array to n with an logical or operation on the whole rows;
        #   ~  - inverts True/False
        wine_dataset = np.array(wine_dataset[~np.isnan(wine_dataset).any(axis=1)])

    # converting quality ratings of wines to binary values
    wine_dataset[:, QUALITY_INDEX][wine_dataset[:, QUALITY_INDEX] <= 5] = 0
    wine_dataset[:, QUALITY_INDEX][wine_dataset[:, QUALITY_INDEX] >= 6] = 1

    # checking for missing/malformed values in the cancer dataset
    if np.argwhere(cancer_dataset == '?').size > 0:
        cancer_dataset = cancer_dataset[~(cancer_dataset == '?').any(axis=1)]
    # converting string-values to int type
    cancer_dataset = cancer_dataset.astype(int)

    print(wine_dataset)
    print(cancer_dataset)


if __name__ == '__main__':
    main()
