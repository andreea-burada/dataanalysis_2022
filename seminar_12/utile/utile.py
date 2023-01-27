import numpy as np
import pandas as pd
import scipy.stats as sts
import pandas.api.types as pdt


def standardize(X):
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xstd = (X - avg) / std
    return Xstd


# Replace NA, NaN, through mean/mode on pandas.DataFrame
def replace_na_df(t):
    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                avg = t[c].mean()
                t[c] = t[c].fillna(avg)
        else:
            if t[c].isna().any():
                mod = t[c].mode()
                t[c] = t[c].fillna(mod[0])


def repalce_na(X):
    avg = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = avg[k_nan[1]]


def toTable(X, col_name=None, index_name=None, tabel=None):
    X_tab = pd.DataFrame(X)
    if col_name is not None:
        X_tab.columns = col_name
    if index_name is not None:
        X_tab.index = index_name
    if tabel is None:
        X_tab.to_csv("tabel.csv")
    else:
        X_tab.to_csv(tabel)
    return X_tab


# t - DataFrame
def coding(t, vars):
    assert isinstance(t, pd.DataFrame), "We need a pandas DataFrame."
    for v in vars:
        t[v] = pd.Categorical(t[v]).codes