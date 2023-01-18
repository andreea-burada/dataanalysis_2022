import numpy as np

def replaceNaN(X):
    avg = np.nanmean(X, axis=0)
    pos = np.where(np.isnan(X))
    #print('NaN Location:', pos)
    X[pos] = avg[pos[1]]
    return X

def standardize(X):
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - avg) / std
    return X_std