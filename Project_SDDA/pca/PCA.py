'''
A class which incapsulates a PCA model
'''
import numpy as np


class PCA:

    def __init__(self, matrix):
        self.X = matrix

        # standardized X
        avg = np.mean(self.X, axis=0)  # have the variables on the columns
        std = np.std(self.X, axis=0)
        self.Xstd = (self.X - avg) / std

        # compute the correlation matrix of X
        # self.Corr = np.corrcoef(self.Xstd, rowvar=False)  # have the variables on the columns

        # compute the of Xstd
        self.Cov = np.cov(self.Xstd, rowvar=False)  # have the variables on the columns

        # compute eigenvalues and eigenvectors for the matrix of variance/covariance
        eigvalues, eigenvectors = np.linalg.eigh(self.Cov)
        print(eigvalues)

        # sort the eigenvalues and eigenvectors in descending order
        k_des = [k for k in reversed(np.argsort(eigvalues))]
        print(k_des)
        self.alpha = eigvalues[k_des]
        print(self.alpha)
        self.a = eigenvectors[:, k_des]

        # regularization of the eigenvectors
        for col in range(self.a.shape[1]):  # len(self.alpha), self.alpha.shape[0]
            minimum = np.min(self.a[:, col])  # the minimum on each column
            maximum = np.max(self.a[:, col])  # the minimum on each column
            if np.abs(minimum) > np.abs(maximum):
                # self.a[:, col] = -self.a[:, col]  # multiplying the column with a scalar (-1)
                self.a[:, col] *= -1

        # compute the principal components
        self.C = self.Xstd @ self.a  # operator @ is overloaded for multiplying matrices

        # compute factor loadings
        self.Rxc = self.a * np.sqrt(self.alpha)

        # compute the scores (standardized principal components)
        self.Scores = self.C / np.sqrt(self.alpha)

        # compute the quality of observations representation on the axis of the principal components
        C2 = self.C * self.C
        C2sums = np.sum(C2, axis=1)  # (4, 6) the axis are counted from the right hand side (need the sums on the rows)
        self.ObsQual = np.transpose(np.transpose(C2) / C2sums)

        # compute the contribution of observation s at variance of PC axis
        self.betha = C2 / (self.alpha * self.X.shape[0])

        # compute the commonalities (retrieving the principle components in the initial, observed variables)
        Rxc2 = self.Rxc * self.Rxc
        self.Common = np.cumsum(Rxc2, axis=1)  # the cumulative summing is done on the rows



    def getXstd(self):
        return self.Xstd

    def getEigenvalues(self):
        return self.alpha

    def getPrinComp(self):
        return self.C

    def getLoadings(self):
        return self.Rxc

    def getScores(self):
        return self.Scores

    def getObsQual(self):
        return self.ObsQual

    def getBetha(self):
        return self.betha

    def getCommon(self):
        return self.Common