import numpy as np

WINE_DATA_PATH = 'resources/winequality-red.csv'
CANCER_DATA_PATH = 'resources/breast-cancer-wisconsin.data'
WINE_QUALITY_INDEX = 11
CANCER_CLASS_INDEX = 10
label_dict = {'benign': 2, 'malignant': 4}

def cleanCancerDataset() -> np.array:
    cancer_dataset = np.genfromtxt(CANCER_DATA_PATH, delimiter=',', skip_header=0)
    # checking for missing/malformed values in the cancer dataset
    if np.argwhere(np.isnan(cancer_dataset)).size > 0:
        cancer_dataset = np.array(cancer_dataset[~np.isnan(cancer_dataset).any(axis=1)])
    # converting cancer classes to binary values
    cancer_dataset[:, CANCER_CLASS_INDEX][cancer_dataset[:, CANCER_CLASS_INDEX] == label_dict['benign']] = 0
    cancer_dataset[:, CANCER_CLASS_INDEX][cancer_dataset[:, CANCER_CLASS_INDEX] == label_dict['malignant']] = 1
    # removing sample code numbers
    cancer_dataset = cancer_dataset[:, 1:]
    # converting string-values to int type
    cancer_dataset = cancer_dataset.astype(int)

    return cancer_dataset

def cleanWineDataset() -> np.array:
    wine_dataset = np.genfromtxt(WINE_DATA_PATH, delimiter=';', skip_header=1)
    
    # checking for missing values in the wine dataset
    # if the resulting array is not empty
    if np.argwhere(np.isnan(wine_dataset)).size > 0:
        # deleting rows with missing values
        #   np.isnan  - returns boolean with True where NaN, and False elsewhere;
        #   .any(axis=1)  - reduces an m*n array to n with an logical or operation on the whole rows;
        #   ~  - inverts True/False
        wine_dataset = np.array(wine_dataset[~np.isnan(wine_dataset).any(axis=1)])

    # converting quality ratings of wines to binary values
    wine_dataset[:, WINE_QUALITY_INDEX][wine_dataset[:, WINE_QUALITY_INDEX] <= 5] = 0
    wine_dataset[:, WINE_QUALITY_INDEX][wine_dataset[:, WINE_QUALITY_INDEX] >= 6] = 1

    return wine_dataset
