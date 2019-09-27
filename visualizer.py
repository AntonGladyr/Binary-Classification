import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from linear_discriminant_analysis import LinearDiscriminantAnalysis

sns.set()


def visualize_multivariate(dataset, columns):
    data = pd.DataFrame(data=dataset, columns=columns)
    sns.pairplot(data, hue=columns[-1])
    plt.show()

def visualize_univariate(dataset, columns):
    #TODO to be fixed
    data = pd.DataFrame(data=dataset, columns=columns)
    f, ax = plt.subplots(1, len(columns) - 1, figsize=(10, 3))
    list = []
    for index, column in enumerate(columns):
        sns.distplot(data[column], bins=10, ax=ax[0])
    plt.show()

def visualize_predictions(X, y, lda: LinearDiscriminantAnalysis):
    #TODO to be fixed
    size_of_map = np.arange(-3, 9, 0.01)
    print(size_of_map.shape)
    xx, yy = np.meshgrid(size_of_map, size_of_map)  # gives a rectangular grid out of
    # input values

    print(xx.shape)

    pred = lda.predict(xx)
    pred = pred.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    #plt.title("Score: %.0f percents" % (clf.score(X, y) * 100))
    plt.show()
