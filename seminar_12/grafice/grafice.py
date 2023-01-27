import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def scatter_obs_grupe(z1, z2, y, labels, zg1, zg2, labels_g):
    f = plt.figure(figsize=(11, 8))
    assert isinstance(f, plt.Figure)
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Observatii si centre de grupe planul axelor z1 si z2", fontsize=14, color='b')
    sb.scatterplot(x=z1, y=z2, hue=y, ax=ax, hue_order=labels_g)
    sb.scatterplot(x=zg1, y=zg2, hue=labels_g, ax=ax, legend=False, marker='s', s=300)
    for i in range(len(labels)):
        ax.text(z1[i], z2[i], labels[i])
    for i in range(len(labels_g)):
        ax.text(zg1[i], zg2[i], labels_g[i], fontsize=24)
    # plt.show()


def distributie_grupe(z, y, g, axa):
    f = plt.figure(figsize=(11, 8))
    assert isinstance(f, plt.Figure)
    sf = f.add_subplot(1, 1, 1)
    assert isinstance(sf, plt.Axes)
    sf.set_title("Distributie pe grupe. Axa " + str(axa + 1), fontsize=14, color='b')
    for v in g:
        sb.kdeplot(data=z[y == v], fill=True, ax=sf, label=v)
    # plt.show()


def show():
    plt.show()
