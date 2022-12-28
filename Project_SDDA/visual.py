import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


# create a correlogram
def correlogram(matrix=None, dec=1, title='Correlogram', valMin=-1, valMax=1):
    plt.figure(title, figsize=(17, 13)) # figsize(X, Y)
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    print(np.round(matrix, dec))
    sb.heatmap(data=np.round(matrix, dec), cmap='bwr', vmin=valMin, vmax=valMax, annot=True)


# create the correlation circle
def corrCircle(matrix=None, V1=0, V2=1, radius=1, dec=2,
               labelX=None, labelY=None, valMin=-1, valMax=1, title='Correlation circle'):
    plt.figure(title, figsize=(8, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t)*radius for t in T]
    Y = [np.sin(t)*radius for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if labelX==None or labelY==None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(xlabel=matrix.columns[V1], fontsize=12, color='k', verticalalignment='top')
            plt.ylabel(ylabel=matrix.columns[V2], fontsize=12, color='k', verticalalignment='bottom')
        else:
            plt.xlabel(xlabel='Var '+str(V1+1), fontsize=12, color='k', verticalalignment='top')
            plt.ylabel(ylabel='Var '+str(V2+1), fontsize=12, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=labelX, fontsize=12, color='k', verticalalignment='top')
        plt.ylabel(ylabel=labelY, fontsize=12, color='k', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, V1], y=matrix[:, V2], c='r', vmin=valMin, vmax=valMax)
        for i in range(matrix.shape[0]):
            # plt.text(x=0.25, y=0.25, s='this is a label')
            plt.text(x=matrix[i, V1], y=matrix[i, V2],
                     s='('+str(round(matrix[i, V1], dec)) +
                         ', ' + str(round(matrix[i, V2], dec)) + ')')

    if isinstance(matrix, pd.DataFrame):
        # plt.text(x=0.25, y=0.25, s='we have a pandas.DataFrame')
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2],
                    c='r', vmin=valMin, vmax=valMax)
        # for i in range(len(matrix.index)):
        for i in range(matrix.values.shape[0]):
            # plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2],
            #          s='(' + str(round(matrix.iloc[i, V1], dec)) +
            #          ', ' + str(round(matrix.iloc[i, V2], dec)) + ')')
            plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2], s=matrix.index[i])


def principalComponents(eigenvalues=None, columns=None, title='Explained variance of the principal components'):
    plt.figure(title, figsize=(15, 11))  # figsize(X, Y)
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel='Principal components', fontsize=12, color='k', verticalalignment='top')
    plt.ylabel(ylabel='Eigenvalues (variance)', fontsize=12, color='k', verticalalignment='bottom')
    if columns==None:
        components = ['C'+str(i+1) for i in range(len(eigenvalues))]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')


def show():
    plt.show()