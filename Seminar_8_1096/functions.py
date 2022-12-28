import numpy as np


def standardize(X):
    if isinstance(X, np.ndarray):
        means = np.mean(a=X, axis=0)  # compute the averages on the columns
        stds = np.std(a=X, axis=0)  # compute the standard deviations on the columns
        Xstd = (X - means) / stds
        return Xstd