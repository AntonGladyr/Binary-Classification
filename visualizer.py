import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()


def visualize(dataset, columns):
    data = pd.DataFrame(data=dataset, columns=columns)
    sns.pairplot(data, hue=columns[-1])
    plt.show()
