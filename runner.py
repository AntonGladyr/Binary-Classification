'''
Main runner

For running details run python3 runner.py -h

Ex:
- python3 runner.py wine_dataset --s 0.9 train LDA
- python3 runner.py wine_dataset validate LDA
'''

#!/usr/bin/env python

import time
import argparse
import numpy as np
from logistic_regression import LogisticRegression as LR
from logistic_regression_vectorized import LogisticRegression as LR_V
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cleaner import cleanWineDataset
from cleaner import cleanCancerDataset
from matplotlib import pyplot as plt
from learning_rates import *

def evaluate_acc(predictions, targets):
    return np.mean(predictions == targets)

def k_fold_runner(model, dataset, k, target_index):
    accuracy_scores = []

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

        accuracy_scores.append( evaluate_acc(model.predict(features_val), targets_val) )
        print('itr %d with accuracy %.2f' % (i, accuracy_scores[-1]*100))

        starting_index = partition_size*(i+1)
    
    print('overall accuracy score of the model = %.2f' % ( (1/len(accuracy_scores))*sum(accuracy_scores)*100 ))

def split_dataset(features, targets, pct):
    train_pct_index = int(pct * len(features))
    X_train, X_val = features[:train_pct_index, :], features[train_pct_index:, :]
    Y_train, Y_val = targets[:train_pct_index], targets[train_pct_index:]
    return X_train, X_val, Y_train, Y_val

def lr_accuracy_plot(X_train, Y_train, X_val, Y_val, lr=[learning_rate_constant6, learning_rate_func1]):
    accuracy = []
    for rate in lr:
        model = LogisticRegression(rate, 1000)
        model.fit(X_train, Y_train)
        accuracy.append((evaluate_acc(model.predict(X_val), Y_val) * 100))
        print('accuracy = %.2f' % (accuracy[-1]))

    y_pos = np.arange(len(lr))
    dd = accuracy # basic inormation
    plt.bar(y_pos, accuracy, width=0.3, alpha=0.9, align='center', color="yrgb")
 
    plt.xticks(y_pos, lr)
    plt.ylabel('accuracy')
    plt.title('learning rate VS accuracy')
 
    plt.show()

def maxitr_accuracy(X_train, Y_train, X_val, Y_val, maxitrs=[100, 1000, 10000, 100000]):    
    accuracy=[]
    for maxitr in maxitrs:
        model = LogisticRegression(learning_rate_func1, maxitr)
        model.fit(X_train,Y_train)
        model.predict(X_val)

        accuracy.append((evaluate_acc(model.predict(X_val), Y_val) * 100))
        print('accuracy = %.2f' % (accuracy[-1]))

    y_pos = np.arange(len(maxitrs))
    dd = accuracy # basic inormation
    plt.bar(y_pos, accuracy, width=0.3, alpha=0.9, align='center', color="yrgb")
 
    plt.xticks(y_pos, maxitrs)
    plt.ylabel('accuracy')
    plt.title('Number of iterations VS accuracy')
 
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COMP551 LDA/LR runner')
    parser.add_argument('dataset', type=str, choices=['wine_dataset', 'cancer_dataset'], help='dataset specifier')
    parser.add_argument('op', type=str, choices=['train', 'validate'], help='operation specifier')
    parser.add_argument('--s', '-split_pct',  type=float, help='percentage of data used for training')

    model_parsers = parser.add_subparsers(help='model specifier')

    lda_p = model_parsers.add_parser('LDA')
    lda_p.add_argument('--k', type=int, default=5, help='k-fold specifier (default: 5)')
    lda_p.set_defaults(which_model='LDA')

    lr_p = model_parsers.add_parser('LR')
    lr_p.add_argument('--k', type=int, default=5, help='k-fold specifier (default: 5)')
    lr_p.add_argument('--lr', '-learning_function', type=str, required=True, help='name of learning function (must be present)')
    lr_p.add_argument('--m', '-method', type=str, choices=['itr', 'threshold'], required=True, help='termination criteria')
    lr_p.add_argument('--t', '-terminating-value', type=float, required=True, help='termination value')
    lr_p.set_defaults(which_model='LR')

    lr_vectorized_p = model_parsers.add_parser('LR_V')
    lr_vectorized_p.add_argument('--k', type=int, default=5, help='k-fold specifier (default: 5)')
    lr_vectorized_p.add_argument('--lr', '-learning_function', type=str, required=True, help='name of learning function (must be present)')
    lr_vectorized_p.add_argument('--m', '-method', type=str, choices=['itr', 'threshold'], required=True, help='termination criteria')
    lr_vectorized_p.add_argument('--t', '-terminating-value', type=float, required=True, help='termination value')
    lr_vectorized_p.set_defaults(which_model='LR_V')

    args = parser.parse_args()
    if args.op == 'train' and args.s is None:
        raise argparse.ArgumentError('split_pct must be set with train operation')

    if args.dataset == 'wine_dataset':
        dataset = cleanWineDataset()
        TARGET_INDEX = 11
    elif args.dataset == 'cancer_dataset':
        dataset = cleanCancerDataset()
        # dataset = np.delete(dataset, 2, axis=1)
        TARGET_INDEX = 9
    np.random.seed(23)
    np.random.shuffle(dataset)
    
    if args.which_model == 'LR':
        if args.m == 'itr':
            model = LR(globals()[args.lr], int(args.t), itr=True)
        elif args.m == 'threshold':
            model = LR(globals()[args.lr], args.t, itr=False)
    elif args.which_model == 'LR_V':
        if args.m == 'itr':
            model = LR_V(globals()[args.lr], int(args.t), itr=True)
        elif args.m == 'threshold':
            model = LR_V(globals()[args.lr], args.t, itr=False)
    elif args.which_model == 'LDA':
        model = LinearDiscriminantAnalysis()

    if args.op == 'train':
        features = dataset[:, :TARGET_INDEX]
        targets = dataset[:, TARGET_INDEX]
        X_train, X_val, Y_train, Y_val = split_dataset(features, targets, args.s)

        start_time = time.time()
        model.fit(X_train, Y_train)
        print('training time elapsed: %.4f seconds' % ( (time.time() - start_time) ))

        print('accuracy = %.2f' % (evaluate_acc(model.predict(X_val), Y_val)*100))
    elif args.op == 'validate':
        k_fold_runner(model, dataset, args.k, TARGET_INDEX)
