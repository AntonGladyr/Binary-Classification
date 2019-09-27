import sys
import numpy as np
from logistic_regression import LogisticRegression
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

        model.fit_itr(features_train, targets_train, 1000)
        print('itr %d with accuracy %f' % (i, np.mean(model.predict(features_val) == targets_val)))
        
        starting_index = partition_size*(i+1)

def evaluate(model, dataset):
    model.predict()

def split_dataset(features, targets, pct):
    train_pct_index = int(pct * len(features))
    X_train, X_val = features[:train_pct_index, :], features[train_pct_index:, :]
    Y_train, Y_val = targets[:train_pct_index], targets[train_pct_index:]
    return X_train, X_val, Y_train, Y_val

accuracy = []
def lr_accuracy(X_train, Y_train, X_val, Y_val):
    # lr = [learning_rate_constant1, learning_rate_constant2, learning_rate_constant3, learning_rate_constant4, 
    # learning_rate_constant5, learning_rate_constant6, learning_rate_constant7, learning_rate_constant8, learning_rate_func1]

    lr = [learning_rate_constant6, learning_rate_func1]
    
    for rate in lr:
        model = LogisticRegression(rate)
        model.fit_itr(X_train, Y_train, 100000)
       
        accuracy.append(np.mean(model.predict(X_val) == Y_val) * 100)
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
        TARGET_INDEX = 'TBD'
    else:
        print('dataset specified not one of (wine_dataset, cancer_dataset)')
        sys.exit(1)


    k_fold_runner(LogisticRegression(learning_rate_func1), dataset, 5)

    # features = dataset[:, :TARGET_INDEX]
    # targets = dataset[:, TARGET_INDEX]
    # X_train, X_val, Y_train, Y_val = split_dataset(features, targets, 0.9)
    
    # # logisticRegression = LogisticRegression(dataset, TARGET_INDEX, 1000, 0.0001)
    # # logisticRegression.fit()
    # # print(logisticRegression.parameters)

    # lr_accuracy(X_train, Y_train, X_val, Y_val)
    sys.exit(0)