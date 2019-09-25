# office hours ask numpy and evaluation


from LogisticRegression_sbakhit import LogisticRegression

NUM_ITERATIONS = 1000

def learning_rate_constant(k):
    return 0.0001

def learning_rate2(k):
    return 1/1+k

def k_fold_runner(logisticRegression: LogisticRegression, k: int):
    #TODO: shuffle
    partition_size = logisticRegression.DATASET_SIZE // k
    starting_index = 0
    for i in range(k):
        dataset = logisticRegression.dataset[starting_index:partition_size*(i+1), :]
        features
        logisticRegression.fit()
        starting_index = partition_size*(i+1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('no dataset specified')
        sys.exit(1)
    if sys.argv[1] == "wine_dataset":
        dataset = cleanWineDataset()
        TARGET_INDEX = 11
    elif sys.argv[1] == "cancer_dataset":
        dataset = cleanCancerDataset()
        TARGET_INDEX = 'TBD'
    else:
        print('dataset specified not one of (wine_dataset, cancer_dataset)')
        sys.exit(1)
    
    logisticRegression = LogisticRegression(dataset, TARGET_INDEX, 1000, 0.0001)
    logisticRegression.fit()
    print(logisticRegression.parameters)
    sys.exit(0)