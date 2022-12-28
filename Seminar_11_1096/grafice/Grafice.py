import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hiclu


def dendrograma(h, etichete, titlu='Clasificare ierarhica', threshold=None):
    plt.figure(figsize=(11, 8))
    plt.title(titlu, fontsize=16, color='k')
    hiclu.dendrogram(Z=h, labels=etichete, leaf_rotation=45)
    plt.axhline(y=threshold, c='r')




def show():
    plt.show()