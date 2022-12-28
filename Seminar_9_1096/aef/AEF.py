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

        return chi2Calc, chi2Tab


