import numpy as np


def inlocuireNAN(X):
    medii = np.nanmean(X, axis=0)  # calcul medii pentru coloanele cu valori NAN
    poz = np.where(np.isnan(X))
    print('Localizare NaN:', poz)
    X[poz] = medii[poz[1]]
    return X
