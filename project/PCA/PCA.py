import numpy as np

'''
Class for implementing PCA - Principal Component Analysis
'''
class PCA:
    def __init__(self, matrix):
        self.X = matrix
        
        # correlation matrix for unstandardized X matrix
        self.R = np.corrcoef(self.X, rowvar=False)  # variables are on columns
        
        # X matrix standardization
        averages = np.mean(self.X, axis=0)      # vars. are on columns
        deviations = np.std(self.X, axis=0)     # computation on columns
        self.X_standardized = (self.X - averages) / deviations
        
        # variance/covariance matrix for standardized X
        self.covariance_matrix = np.cov(m=self.X_standardized, rowvar=False)
        
        # computation for eigenvalues and eigenvectors for variance/covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        #print(eigenvalues)
        
        # sort in descending order the eigenvalues and eigenvectors
        k_desc = [k for k in reversed(np.argsort(eigenvalues))]
        #print(k_desc)
        
        self.alpha = eigenvalues[k_desc]
        self.a = eigenvectors[:, k_desc]
        #print(self.alpha)
        
        # regularization of eigenvectors
        for col in range(self.a.shape[1]):
            minim = np.min(self.a[:, col])  # min and max computation for each eigenvector
            maxim = np.max(self.a[:, col])
            if np.abs(minim) > np.abs(maxim):
                self.a[:, col] *= -1
                
        # prinicipal components computation
        self.C = self.X_standardized @ self.a
        
        # factor loadings matrix
        # -> correlation between the initial variables and the principal components
        self.Rxc = self.a * np.sqrt(self.alpha)
        
        # principal components standardized -> scores
        self.scores = self.C / np.sqrt(self.alpha)
        
        # quality representing observations on the principal components axis
        C2 = self.C * self.C
        C2_sum = np.sum(C2, axis=1)     # sum on the rows, for each observation
        self.calculated_observations = np.transpose(np.transpose(C2) / C2_sum)
        
        # observation contribution for principal components variance
        self.beta = C2 / (self.alpha * self.X.shape[0])
        
        # common factors
        Rxc2 = self.Rxc * self.Rxc
        self.common = np.cumsum(Rxc2, axis=1)
        
    def get_corr(self):
        return self.R
    
    def get_X_standardized(self):
        return self.X_standardized
    
    def get_eigenvalues(self):
        return self.alpha
    
    def get_principal_components(self):
        return self.C
    
    def get_Rxc(self):
        return self.Rxc
    
    def get_scores(self):
        return self.scores
    
    def get_calculated_observations(self):
        return self.calculated_observations
    
    def get_beta(self):
        return self.beta
    
    def get_common(self):
        return self.common

    def get_covariance_matrix(self):
        return self.covariance_matrix
        