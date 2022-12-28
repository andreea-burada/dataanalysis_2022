import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


def correlogram(matrix=None, dec=1, title='Correlogram', min_val=-1, max_val=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), cmap='RdBu', vmin=min_val, vmax=max_val, annot=True)


def correlation_circle(matrix=None, X1=0, X2=1, raza=1, dec=2, title='Correlation Circle', valMin=-1, valMax=1, labelX=None, labelY=None):
    plt.figure(title, figsize=(8, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    # coordinates of points on the circle
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t)*raza for t in T]
    Y = [np.sin(t)*raza for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if labelX==None or labelY==None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(xlabel=matrix.columns[X1], fontsize=14, color='b', verticalalignment='top')
            plt.ylabel(ylabel=matrix.columns[X2], fontsize=14, color='b', verticalalignment='bottom')
        else:
            plt.xlabel(xlabel='Var ' + str(X1 + 1), fontsize=14, color='b', verticalalignment='top')
            plt.ylabel(ylabel='Var ' + str(X2 + 1), fontsize=14, color='b', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=labelX, fontsize=14, color='b', verticalalignment='top')
        plt.ylabel(ylabel=labelY, fontsize=14, color='b', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, X1], y=matrix[:, X2], c='r', vmin=valMin, vmax=valMax)
        for i in range(matrix.shape[0]):
            plt.text(x=matrix[i, X1], y=matrix[i, X2], s='(' +
                    str(np.round(matrix[i, X1], dec)) + ', ' +
                    str(np.round(matrix[i, X2], dec)) + ')')

    if isinstance(matrix, pd.DataFrame):
        plt.scatter(x=matrix.iloc[:, X1], y=matrix.iloc[:, X2], c='r', vmin=valMin, vmax=valMax)
        for i in range(matrix.values.shape[0]):
            plt.text(x=matrix.iloc[i, X1], y=matrix.iloc[i, X2], s=matrix.index[i])


def principal_components(eigenvalues=None, title='Variance explained by principal components', labelX='Principal components', labelY='Explained variance - eigenvalues'):
    plt.figure(title, figsize=(11, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=labelX, fontsize=14, color='b', verticalalignment='top')
    plt.ylabel(ylabel=labelY, fontsize=14, color='b', verticalalignment='bottom')
    componente = ['C'+str(i+1) for i in range(len(eigenvalues))]
    plt.axhline(y=1, color='r')
    plt.plot(componente, eigenvalues, 'bo-')


def scatterPlot(matrix=None, title='Scatter Plot', labelX='Variables', labelY='Observations'):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=labelX, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(ylabel=labelY, fontsize=14, color='k', verticalalignment='bottom')

    plt.scatter(x=matrix.iloc[:, 0].values, y=matrix.index[:])


def show():
    plt.show()
