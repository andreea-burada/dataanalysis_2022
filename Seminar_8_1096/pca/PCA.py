'''
A class for implementing PCA (Principal Components Analysis)
'''

import numpy as np

class PCA:
    def __init__(self, X):
        self.X = X

        # compute the variance-covariance matrix of X
        self.Cov = np.cov(m=X, rowvar=False)  # we hav ethe variables on the columns

        # compute eigenvalues and eigenvector for
        # variance-covariance matrix
        values, vectors = np.linalg.eigh(a=self.Cov)
        print(values)
        print(vectors.shape)
        # sort in descending order the eigenvalues and the eigenvectors
        k_desc = [k for k in reversed(np.argsort(values))]
        print(k_desc)
        self.alpha = values[k_desc]
        self.a = vectors[:, k_desc]
        print(self.alpha)

        # compute the principal components
        self.C = self.X @ self.a  # operator overloaded in numpy
        # for multiplying rectangular matrices
        # self.C = np.matmul(x1=self.X, x2=self.a)

        # compute the factor loadings (the correlation between the
        # observed variables and the principal components)
        self.Rxc = self.a * np.sqrt(self.alpha)


    def getCov(self):
        return self.Cov

    def getEigenValues(self):
        return self.alpha

    def getPrincipalComponents(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScores(self):
        return self.C / np.sqrt(self.alpha)



