import numpy as np
import PCA.PCA as pca
import scipy.stats as sts

'''
Class for implementing EFA - Exploratory Factor Analysis
'''
class EFA:
    '''
    matrix - np.ndarray
    '''
    def __init__(self, matrix):
        self.X = matrix

        # PCA model default
        pca_model = pca.PCA(self.X)
        self.X_standardized = pca_model.get_X_standardized()
        self.Corr = pca_model.get_corr()
        self.eigenvalues = pca_model.get_eigenvalues()
        self.scores = pca_model.get_scores()
        self.calculated_observations = pca_model.calculated_observations

    def get_X_standardized(self):
        return self.X_standardized

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_scores(self):
        return self.scores

    def get_calculated_observations(self):
        return self.calculated_observations

    def BartlettTest(self, loadings, epsilon):
        n = self.X.shape[0]
        m, q = np.shape(loadings)
        V = self.Corr

        # Ψ (psi) - diagonal matrix
        psi = np.diag(epsilon)

        # V estimated
        V_estim = loadings @ np.transpose(loadings) + psi

        # identity matrix estimated
        I_estim = np.linalg.inv(V_estim) @ V

        det_I = np.linalg.det(I_estim)

        if det_I > 0:
            trace_I = np.trace(I_estim)
            chi_square_calc = (n - 1 - (2 * m + 4 * q - 5) / 6) * (trace_I - np.log(det_I) - m)
            no_degrees = ((m - q) ** 2 - m - q) / 2

            chi_square_table = 1 - sts.chi2.cdf(chi_square_calc, no_degrees)
        else:
            chi_square_calc, chi_square_table = np.nan, np.nan

        return chi_square_calc, chi_square_table