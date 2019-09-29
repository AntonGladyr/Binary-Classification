import sys
from logistic_regression_vectorized import LogisticRegression
from runner import split_dataset, evaluate_acc
from cleaner import cleanWineDataset, cleanCancerDataset
import numpy as np
import matplotlib.pyplot as plt
from learning_rates import *

def annot_max(points, pos, ax=None):
    ymax = max(points, key=float)
    xmax = points[ymax]

    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(pos[0],pos[1]), **kw)
    return ax

def plot_itr_acc():
    wine_points = {}
    itr_list_wine = []
    acc_list_wine = []

    cancer_points = {}
    itr_list_cancer = []
    acc_list_cancer = []

    for i in range(0, 100000, 1000):
        model = LogisticRegression(learning_rate_func1, i, itr=True)
        
        model.fit(X_train_wine, Y_train_wine)
        itr_list_wine.append(i)
        acc_list_wine.append(evaluate_acc(model.predict(X_val_wine), Y_val_wine))
        wine_points[evaluate_acc(model.predict(X_val_wine), Y_val_wine)] = i

        model.fit(X_train_cancer, Y_train_cancer)
        itr_list_cancer.append(i)
        acc_list_cancer.append(evaluate_acc(model.predict(X_val_cancer), Y_val_cancer))
        cancer_points[evaluate_acc(model.predict(X_val_cancer), Y_val_cancer)] = i
    
    plt.plot(itr_list_wine, acc_list_wine, label='wine dataset')
    plt.plot(itr_list_cancer, acc_list_cancer, label='cancer dataset')
    annot_max(wine_points, [0.9, 0.1])
    annot_max(cancer_points, [0.5, 0.2])
    plt.legend(loc='upper left')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Optimal Number of Iterations Based on Accuracy')
    plt.show()

def plot_lr_acc(maxitr, dataset=False):
    points = {}
    itr_list = []
    acc_list = []

    if dataset:
        # cancer
        X_train, X_val, Y_train, Y_val = X_train_cancer, X_val_cancer, Y_train_cancer, Y_val_cancer
    else:
        # wine
        X_train, X_val, Y_train, Y_val = X_train_wine, X_val_wine, Y_train_wine, Y_val_wine

    lrs = ['learning_rate_constant1', 'learning_rate_constant2', 'learning_rate_constant3',
            'learning_rate_constant4', 'learning_rate_constant5', 'learning_rate_constant6'
            , 'learning_rate_constant7', 'learning_rate_constant8', 'learning_rate_func1']
    for lr in lrs:
        model = LogisticRegression(globals()[lr], maxitr, itr=True)
        
        model.fit(X_train, Y_train)
        itr_list.append(lr)
        acc_list.append(evaluate_acc(model.predict(X_val), Y_val))
        points[evaluate_acc(model.predict(X_val), Y_val)] = lr

    y_pos = np.arange(len(lrs))
    plt.bar(y_pos, acc_list, width=0.3, alpha=0.9, align='center', color="yrgb")
    plt.xticks(y_pos, lrs)
    
    plt.legend(loc='upper left')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Optimal Learning Rate Based on Accuracy')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    np.random.seed(23)

    wine_dataset = cleanWineDataset()
    TARGET_index_wine = 11

    np.random.shuffle(wine_dataset)

    features_wine = wine_dataset[:, :TARGET_index_wine]
    targets_wine = wine_dataset[:, TARGET_index_wine]
    X_train_wine, X_val_wine, Y_train_wine, Y_val_wine = split_dataset(features_wine, targets_wine, 0.9)

    cancer_dataset = cleanCancerDataset()
    TARGET_index_cancer = 9

    np.random.shuffle(cancer_dataset)

    features_cancer = cancer_dataset[:, :TARGET_index_cancer]
    targets_cancer = cancer_dataset[:, TARGET_index_cancer]
    X_train_cancer, X_val_cancer, Y_train_cancer, Y_val_cancer = split_dataset(features_cancer, targets_cancer, 0.9)
    
    op = sys.argv[1]
    if op == 'itr':
        plot_itr_acc()
    elif op == 'lr':
        if len(sys.argv) < 3:
            sys.exit(1)

        dataset = sys.argv[2]
        if dataset == 'wine_dataset':
            optimal_itr = 99000
            plot_lr_acc(optimal_itr, dataset=False)
        elif dataset == 'cancer_dataset':
            optimal_itr = 4000
            plot_lr_acc(optimal_itr, dataset=True)
    else:
        sys.exit(1)
    
    sys.exit(0)