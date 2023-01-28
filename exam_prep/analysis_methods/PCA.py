'''
imports
'''
# external
import pandas
import numpy as np
# internal
from utils.utils import loadJSON
from utils.utils import replaceNaN
from utils.utils import standardize
import analysis_methods.PCA_graphs as PCA_graphs

'''
PCA - Principal Component Analysis class
input: X - non-standardized matrix of values
output: many different things -> see attributes/getters
'''
class PCA_class:
    def __init__(self, X):
        self.X = X                  # X is non-standardized!!!
        self.X_std = None
        self.R = None               # correlation matrix for non-standardized X
        self.R_std = None           # correlation matrix for standardized X
        self.CovarianceMatrix = None
        self.eigenvalues = None     
        self.eigenvectors = None    
        self.alpha = None           # descending order eigenvalues; standard deviations of princ. comp.
        self.a = None               # descending order eigenvectors
        self.C = None               # principal components
        self.scores = None          # scores or standardized principal components (C_std)
        self.R_xc = None            # factor loadings or correlation factors
        self.obs_qual = None        # representation quality of observations on the principal components axis
        self.betha = None           # observation contribution to axis variance
        self.commonalities = None   # commonalities of principal components found in the initial and casual (cauzale) variables
    
    '''
    process initial matrix and generate all data needed for the PCA model
    '''
    def process(self):
        # compute correlation matrix for non-standardized X
        self.R = np.corrcoef(self.X, rowvar=False)    # rowvar = False -> variables are on COLUMNS
        
        # standardize X
        self.X_std = standardize(self.X)
        
        # compute correlation matrix for standardized X
        self.R_std = np.corrcoef(self.X_std, rowvar=False)
        
        # compute covariance matrix using the standardized X
        self.CovarianceMatrix = np.cov(self.X_std, rowvar=False)
        
        # generate eigenvalues and eigenvectors using the covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.CovarianceMatrix)
        
        # reverse order the eigenvals and eigenvect -> alpha, a
        k_desc = [k for k in reversed(np.argsort(self.eigenvalues))]
        self.alpha = self.eigenvalues[k_desc]
        self.a = self.eigenvectors[:, k_desc]
        
        # alpha and a need to be regularized
        # regularization = if in an eigenvector the absolute value of minimum is greater than the value of the maximum
        #                  then it is statistically more useful to inverse the values of the eigenvector
        for j in range(len(self.alpha)):
            # get eigenvector of eigenvalue on position j
            current_eigenvector = self.a[:, j]
            
            # get maximum and minimum of eigenvector for eigenvalue j
            maxi = np.max(current_eigenvector)
            mini = np.min(current_eigenvector)
            
            if abs(mini) > abs(maxi):
                # reverse the sign of current_eigenvector
                self.a[:, j] *= -1
                
        # compute the principal components for X standardized using eigenvectors
        # the principal components matrix is computed by multiplying the X stand. with the eigenvectors
        # ADA_EN_lecture_2.pdf - page 4:
        #   the link between causal variables X and the principal components C is:
        #       C_k = X ∙ a_k
        self.C = self.X_std @ self.a
        
        # C_std or scores computation
        # dividing the princ. comp. matrix by square root of eigenvalues
        # ADA_EN_lecture_3.pdf - page 11
        self.scores = self.C / np.sqrt(self.alpha)
        
        # compute the factor loadings - correlation between principal components and the initial variables
        # ADA_EN_lecture_3.pdf - page 16:
        #   R_r = a_r ∙ sqrt(alpha_r) - the correlation coefficient vector between X and C_r
        self.R_xc = self.a * np.sqrt(self.alpha)
        
        # commonalities
        # ADA_EN_lecture_3.pdf - page 17
        #   sum squared of factor loadings
        self.commonalities = np.cumsum(self.R_xc * self.R_xc, axis=1)   # on lines
        
        # quality of observations / quality of points representation
        # ADA_EN_lecture_3.pdf - page 13:
        #   because in the sum k is the index of the column -> the sum is computed on the lines
        C_square = self.C * self.C
        C_square_sum = np.sum(C_square, axis=1)
        self.obs_qual = np.transpose(np.transpose(C_square) / C_square_sum)
        
        # observation contributions to axis variances
        # ADA_EN_lecture_3.pdf - page 14
        n = np.shape(self.X)[0]     # n - number of observations
        self.betha = C_square / (n * self.alpha)
        
        
    '''
    Getters
    '''
    def getX_std(self):
        return self.X_std   
    def getR(self):
        return self.R 
    def getR_std(self):
        return self.R_std
    def getCovarianceMatrix(self):
        return self.CovarianceMatrix
    def getEigenvalues(self):
        return self.eigenvalues
    def getEigenvectors(self):
        return self.eigenvectors
    def getAlpha(self):
        return self.alpha
    def getA(self):
        return self.a
    def getPrincipalComponents(self):
        return self.C
    def getScores(self):
        return self.scores
    def getR_xc(self):
        return self.R_xc
    def getCommonalities(self):
        return self.commonalities
    def getObservationQualities(self):
        return self.obs_qual
    def getObservationContributions(self):
        return self.betha
       
       
        
def main():
    # paths_json = loadJSON()
    # dataset_df = pandas.read_excel(io=paths_json["imports"]["PCA"], index_col=0, na_values=':')
    teritorial_df = pandas.read_excel("./dataIN/dataset_teritorial_PCA.xlsx", index_col=0)
    
    X = teritorial_df.iloc[:, 1:].values
    #print(X)
    
    # because X contains empty values -> replace NaN
    # X = replaceNaN(X)
    
    X_df = pandas.DataFrame(data=X, index=teritorial_df.index.values, columns=teritorial_df.columns[1:].values)
    print(X_df)
    obs_name = list(X_df.index.values)
    var_name = list(X_df.columns.values)
    
    model_PCA = PCA_class(X)
    model_PCA.process()
    
    # print(np.shape(model_PCA.getCovarianceMatrix()))
    # print(model_PCA.getEigenvalues())
    # print(np.shape(model_PCA.getEigenvectors()))
    
    alpha = model_PCA.getAlpha()
    PCA_graphs.eigenvaluesGraph(alpha=alpha)
    
    # draw correlogram of matrix of correlation
    R = model_PCA.getR()
    R_df = pandas.DataFrame(data=R, index=var_name, columns=var_name)
    PCA_graphs.correlogram(matrix=R_df, title='Correlogram of Correlation Matrix for X non-standardized')
    
    # draw correlogram of matrix of correlation
    R_std = model_PCA.getR_std()
    R_std_df = pandas.DataFrame(data=R_std, index=var_name, columns=var_name)
    PCA_graphs.correlogram(matrix=R_std_df, title='Correlogram of Correlation Matrix for X standardized')
    
    # draw correlogram of communalities
    common = model_PCA.getCommonalities()
    common_df = pandas.DataFrame(data=common, index=var_name, columns=['C' + str(i + 1) for i in range(len(var_name))])
    PCA_graphs.correlogram(matrix=common_df, title='Correlogram of Commonalities')