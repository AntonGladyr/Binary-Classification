import sys
import numpy as np
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from cleaner import cleanWineDataset
from cleaner import cleanCancerDataset

# Different learning rates
def learning_rate_constant1(k):
    return 1.0

def learning_rate_constant2(k):
    return 0.01

def learning_rate_constant3(k):
    return 0.001

def learning_rate_constant4(k):
    return 0.0001

def learning_rate_constant5(k):
    return 0.00001

def learning_rate_constant6(k):
    return 0.000001

def learning_rate_constant7(k):
    return 0.0000001

def learning_rate_constant8(k):
    return 0.00000001

def learning_rate_func1(k):
    return 1/(1+k)

def evaluate_acc(self, predictions, targets):
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

def lr_accuracy(X_train, Y_train, X_val, Y_val):
    # lr = [learning_rate_constant1, learning_rate_constant2, learning_rate_constant3, learning_rate_constant4, 
    # learning_rate_constant5, learning_rate_constant6, learning_rate_constant7, learning_rate_constant8, learning_rate_func1]

    lr = [learning_rate_constant6, learning_rate_func1]

    accuracy = []
    for rate in lr:
        model = LogisticRegression(rate)
        model.fit_itr(X_train, Y_train, 100000)
        accuracy.append(evaluate_acc((model.predict(X_val), Y_val) * 100)
        print('accuracy = %f' % (accuracy[-1]))

    y_pos = np.arange(len(lr))
    dd = accuracy # basic inormation
    plt.bar(y_pos,accuracy,width=0.3, alpha=0.9,align='center',color="yrgb")
 
    plt.xticks(y_pos, lr)
    plt.ylabel('accuracy')
    plt.title('learning rate VS accuracy')
 
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('no dataset specified')
        sys.exit(1)

    dataset = None
    TARGET_INDEX = None
    if sys.argv[1] == "wine_dataset":
        dataset = cleanWineDataset()
        TARGET_INDEX = 11
    elif sys.argv[1] == "cancer_dataset":
        dataset = cleanCancerDataset()
        TARGET_INDEX = 9
    else:
        print('ERROR: dataset specified not. Accept one of (wine_dataset, cancer_dataset)')
        sys.exit(1)
    
    features = dataset[:, :TARGET_INDEX]
    targets = dataset[:, TARGET_INDEX]
    
    if sys.argv[2] == 'train' or sys.argv[2] == 'validate':
        split_pct = float(sys.argv[4])
        X_train, X_val, Y_train, Y_val = split_dataset(features, targets, split_pct)

        if sys.argv[3] == 'LR':
            lr_func = globals()[sys.argv[5]]
            method = sys.argv[6]
            condition = float(sys.argv[7])

            model = None
            if method == 'itr':
                model = LogisticRegression(lr_func, int(condition), itr=True)
            elif method == 'threshold':
                model = LogisticRegression(learning_rate_func1, condition, itr=False)
            else:
                print('ERROR: invalid method. Accept one of (itr, threshold)')
                sys.exit(1)
            
            if sys.argv[2] == 'train':
                model.fit(X_train, Y_train)
                print(evaluate_acc(model.predict(X_val), Y_val))
            else:
                k = int(sys.argv[-1])
                k_fold_runner(model, dataset, k, TARGET_INDEX)
        elif sys.argv[3] == 'LDA':
            model = LinearDiscriminantAnalysis()
            if sys.argv[2] == 'train':
                model.fit(X_train, Y_train)
                print(evaluate_acc(model.predict(X_val), Y_val))
            else:
                k = int(sys.argv[-1])
                k_fold_runner(model, dataset, k, TARGET_INDEX)
        else:
            print('ERROR: invalid model. Accept one of (LDA, LR)')
            sys.exit(1)
    else:
        print('ERROR: invalid operation. Accept one of (train, validate)')
        sys.exit(1)

    # lr_accuracy(X_train, Y_train, X_val, Y_val)
    sys.exit(0)
