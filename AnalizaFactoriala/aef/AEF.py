'''
clasa care incapsuleaza o implementare de AEF
'''
import numpy as np
import acp.ACP as acp
import scipy.stats as sts


class AEF:

    def __init__(self, matrice):  # parametru asteptat este un numpy.ndarray
        self.X = matrice

        # intantiere model ACP
        acpModel = acp.ACP(self.X)
        self.Xstd = acpModel.getXstd()
        self.Corr = acpModel.getCorr()
        self.ValProp = acpModel.getValProp()
        self.Scoruri = acpModel.getScoruri()
        self.CalObs = acpModel.CalObs

    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.ValProp

    def getScoruri(self):
        return self.Scoruri

    def getCalObs(self):
        return self.CalObs

    def calculTestBartlett(self, loadings, epsilon):
        #  TO DO
        n = self.X.shape[0]
        m, q = np.shape(loadings)
        # print(n, m, q)
        V = self.Corr

        # matrice diagonala psi
        psi = np.diag(epsilon)

        # V estimat
        V_estim = loadings @ np.transpose(loadings) + psi

        # matrice identitiate estimata
        I_estim = np.linalg.inv(V_estim) @ V

        det_I = np.linalg.det(I_estim)

        if det_I > 0:
            trace_I = np.trace(I_estim)
            chi_square_calc = (n - 1 - (2 * m + 4 * q - 5) / 6) * (trace_I - np.log(det_I) - m)
            nr_grade = ((m - q) ** 2 - m - q) / 2

            chi_square_table = 1 - sts.chi2.cdf(chi_square_calc, nr_grade)
        else:
            chi_square_calc, chi_square_table = np.nan, np.nan

        return chi_square_calc, chi_square_table


