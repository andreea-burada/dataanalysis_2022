import numpy as np


def replaceNAN(X):
    avgs = np.nanmean(a=X, axis=0)  # the means are computed on the columns
    pos = np.where(np.isnan(X))
    print(pos)
    X[pos] = avgs[pos[1]]
    return X


