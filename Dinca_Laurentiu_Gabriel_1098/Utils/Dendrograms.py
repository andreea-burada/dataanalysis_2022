import scipy.cluster.hierarchy as hic
import matplotlib.pyplot as plt


def dendrogram(h, labels, title='Hierarchical Classification', threshold=None):
    plt.figure(figsize=(16, 9))
    plt.title(title, fontsize=12, color='k')
    hic.dendrogram(h, labels=labels, leaf_rotation=90)
    if threshold:
        plt.axhline(threshold, c='r')
        print('--------------Printing the threshold-------------------')
        print(threshold)
        print('-------------------------------------------------------')


def display():
    plt.show()
