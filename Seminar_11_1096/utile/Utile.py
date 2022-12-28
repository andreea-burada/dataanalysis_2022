import numpy as np
import pandas as pd


def inlocuireNaN(X):
    medie = np.nanmean(X, axis=0)
    pozitie = np.where(np.isnan(X))
    X[pozitie] = medie[pozitie[1]]
    return X

def standizare(X):
    medie = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xstd = (X - medie) / std
    return Xstd

def calculThreshold(h):  # determinare valoare partie optima (de maxima stabilitate)
#     # TODO

     nrJonctiuni = h.shape[0]  # np.shape(h)[0]
     dist_1 = h[1:,2]
     dist_2 = h[:nrJonctiuni-1,2]
     diff = dist_1 - dist_2
     print(diff)
     jonctiuneDifMax = np.argmax(diff)
     print(jonctiuneDifMax)
     threshold = (h[jonctiuneDifMax,2] + h[jonctiuneDifMax, 2]) / 2
     return threshold, nrJonctiuni, jonctiuneDifMax
#
# def determinareClustere(h, jonctiuneDifMax):  # determinare clustere din partitia de maxima stabilitate
#     # TODO

