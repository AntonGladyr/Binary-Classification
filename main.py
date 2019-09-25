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
from linear_discriminant_analysis import LinearDiscriminantAnalysis


def main():
    wine_dataset = cleaner.cleanWineDataset()
    cancer_dataset = cleaner.cleanCancerDataset()
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
