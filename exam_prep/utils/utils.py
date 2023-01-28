import json
import numpy as np

def loadJSON():
    file = open("paths.json")
    toReturn = json.load(file)
    file.close()
    return toReturn

def replaceNaN(matrix):
    averages = np.nanmean(a=matrix, axis=0)     # axis 0 - columns
    positions = np.where(np.isnan(matrix))
    #print('positions NaN', positions)
    matrix[positions] = averages[positions[1]]
    return matrix

def standardize(X):
    # averages on columns
    averages = np.mean(X, axis=0)   # axis 0 - columns
    # standard deviation on columns
    stand_dev = np.std(X, axis=0)
    
    return (X - averages) / stand_dev