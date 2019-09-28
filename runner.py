#!/usr/bin/env python

import sys
import argparse
import numpy as np
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cleaner import cleanWineDataset
from cleaner import cleanCancerDataset
from matplotlib import pyplot as plt
from learning_rates import *

def evaluate_acc(predictions, targets):
    return np.mean(predictions == targets)

def k_fold_runner(model, dataset, k, target_index):
    np.random.shuffle(dataset)

    partition_size = dataset.shape[0] // k
    starting_index = 0
    for i in range(k):
        dataset_val = dataset[starting_index:partition_size*(i+1), :]
        dataset_train = np.delete(dataset, np.s_[starting_index:partition_size*(i+1)], axis=0)

        features_train = dataset_train[:, :target_index]
        targets_train = dataset_train[:, target_index]

        features_val = dataset_val[:, :target_index]
        targets_val = dataset_val[:, target_index]

        model.fit(features_train, targets_train)
        print('itr %d with accuracy %f' % (i, evaluate_acc(model.predict(features_val), targets_val)))

        starting_index = partition_size*(i+1)

def split_dataset(features, targets, pct):
    train_pct_index = int(pct * len(features))
    X_train, X_val = features[:train_pct_index, :], features[train_pct_index:, :]
    Y_train, Y_val = targets[:train_pct_index], targets[train_pct_index:]
    return X_train, X_val, Y_train, Y_val

def lr_accuracy_plot(X_train, Y_train, X_val, Y_val):
    # lr = [learning_rate_constant1, learning_rate_constant2, learning_rate_constant3, learning_rate_constant4, 
    # learning_rate_constant5, learning_rate_constant6, learning_rate_constant7, learning_rate_constant8, learning_rate_func1]

    lr = [learning_rate_constant6, learning_rate_func1]

    accuracy = []
    for rate in lr:
        model = LogisticRegression(rate, 1000)
        model.fit(X_train, Y_train)
        accuracy.append((evaluate_acc(model.predict(X_val), Y_val) * 100))
        print('accuracy = %f' % (accuracy[-1]))

    y_pos = np.arange(len(lr))
    dd = accuracy # basic inormation
    plt.bar(y_pos, accuracy, width=0.3, alpha=0.9, align='center', color="yrgb")
 
    plt.xticks(y_pos, lr)
    plt.ylabel('accuracy')
    plt.title('learning rate VS accuracy')
 
    plt.show()

def maxitr_accuracy(X_train, Y_train, X_val, Y_val):
    itrs = [1, 10, 1000, 10000, 100000, 1000000]
    
    accuracy=[]
    for maxitr in itrs:
        model = LogisticRegression(learning_rate_func1, maxitr)
        model.fit(X_train,Y_train)
        model.predict(X_val)

        accuracy.append((evaluate_acc(model.predict(X_val), Y_val) * 100))
        print('accuracy = %f' % (accuracy[-1]))

    y_pos = np.arange(len(itrs))
    dd = accuracy # basic inormation
    plt.bar(y_pos, accuracy, width=0.3, alpha=0.9, align='center', color="yrgb")
 
    plt.xticks(y_pos, itrs)
    plt.ylabel('accuracy')
    plt.title('Number of iterations VS accuracy')
 
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COMP551 LDA/LR runner')
    parser.add_argument('dataset', type=str, metavar='', choices=['wine_dataset', 'cancer_dataset'], help='dataset specifier')
    parser.add_argument('op', type=str, choices=['train', 'validate'], help='operation specifier')
    parser.add_argument('--s', '-split_pct',  type=float, metavar='', help='percentage of data used for training')

    model_parsers = parser.add_subparsers(help='model specifier')

    lda_p = model_parsers.add_parser('LDA')
    lda_p.add_argument('--k', type=int, metavar='', default=5, help='k-fold specifier (default: 5)')
    lda_p.set_defaults(which_model='LDA')

    lr_p = model_parsers.add_parser('LR')
    lr_p.add_argument('--k', type=int, metavar='', default=5, help='k-fold specifier (default: 5)')
    lr_p.add_argument('--lr', '-learning_function', type=str, metavar='', required=True, help='name of learning function (must be present)')
    lr_p.add_argument('--m', '-method', type=str, metavar='', choices=['itr', 'threshold'], required=True, help='termination criteria')
    lr_p.add_argument('--t', '-terminating-value', type=float, metavar='', required=True, help='termination value')
    lr_p.set_defaults(which_model='LR')

    args = parser.parse_args()

    if args.dataset == 'wine_dataset':
        dataset = cleanWineDataset()
        TARGET_INDEX = 11
    elif args.dataset == 'cancer_dataset':
        dataset = cleanCancerDataset()
        TARGET_INDEX = 9
    
    if args.which_model == 'LR':
        if args.m == 'itr':
            model = LogisticRegression(globals()[args.lr], int(args.t), itr=True)
        elif args.m == 'threshold':
            model = LogisticRegression(globals()[args.lr], args.t, itr=False)
    elif args.which_model == 'LDA':
        model = LinearDiscriminantAnalysis()

    if args.op == 'train':
        features = dataset[:, :TARGET_INDEX]
        targets = dataset[:, TARGET_INDEX]
        X_train, X_val, Y_train, Y_val = split_dataset(features, targets, args.s)

        model.fit(X_train, Y_train)
        print(evaluate_acc(model.predict(X_val), Y_val))
    elif args.op == 'validate':
        k_fold_runner(model, dataset, args.k, TARGET_INDEX)

    sys.exit(0)
