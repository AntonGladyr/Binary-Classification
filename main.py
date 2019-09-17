#!/usr/bin/env python
"""
COMP 551
MiniProject 1

This mini project implements two linear classification techniques — logistic regression and linear discriminant
analysis (LDA) as well as runs these two algorithms on two distinct datasets.

Authors:
            Anton Gladyr    anton.gladyr@mail.mcgill.ca
            Saleh Bakhit    saleh.bakhit@mail.mcgill.ca
            Vasu Khanna     TBA

Created on 16 Sep 2019
"""

import numpy as np
from pandas import read_csv as read

WINE_DATA_PATH = "./resources/winequality-red.csv"
CANCER_DATA_PATH = "./resources/breast-cancer-wisconsin.data"


def main():
    wine_data = read(WINE_DATA_PATH, delimiter=",")
    cancer_data = read(CANCER_DATA_PATH, delimiter=",")
    print(wine_data.head())


if __name__ == '__main__':
    main()
