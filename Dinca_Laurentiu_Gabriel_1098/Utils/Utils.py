import numpy as np
import pandas as pd


def replaceNaN(X):
    avg = np.nanmean(X, axis=0)
    print(avg)
    posNaN = np.where(np.isnan(X))
    X[posNaN] = avg[posNaN[1]]
    return X


def standardize(X):
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xstd = (X - avg) / std
    return Xstd


# Computing the value of the threshold
def threshold(h):
    # the number of maximum junctions
    m = np.shape(h)[0]  # computing the total number of rows
    dist_1 = h[1:m, 2]  # computing the distances, besides the first row -> for the second column only
    dist_2 = h[0:m - 1, 2]  # computing the distances, besides the last row -> for the second column only
    diff = dist_1 - dist_2
    j = np.argmax(diff)
    threshold = (h[j, 2] + h[j + 1, 2]) / 2
    return threshold, j, m


# Determine the clusters of the maximum stability partition
def clusters(h, k):
    n = np.shape(h)[0] + 1
    g = np.arange(0, n)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    cat = pd.Categorical(g)
    print('--------------------------')
    print(cat)
    print('--------------------------')
    return ['C'+str(i) for i in cat.codes], cat.codes

