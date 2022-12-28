import pandas as pd
import pca.PCA as pca
import visual as vi
import numpy as np


table = pd.read_csv('dataIN/Datas.csv', index_col=0)
print(table)

# create a list of variables
varName = list(table.columns)[1:]
# create a list observation names
obsName = list(table.index)

# extract the matrix containing the values of the observed variables
X = table[varName].values
print(X)

# create an instance of PCA class
pca_model = pca.PCA(X)
Xstd = pca_model.getXstd()
# save in a SCV file X standardized
Xstd_df = pd.DataFrame(data=Xstd, index=obsName, columns=varName)
Xstd_df.to_csv('dataOUT/Xstd.csv')

# create the graphic for the explained variance by the principal components
vi.principalComponents(eigenvalues=pca_model.getEigenvalues())
# vi.show()

# save principal components into a CSV file
prinComp = pca_model.getPrinComp()
prinComp_df = pd.DataFrame(data=prinComp, index=obsName, columns=('C'+str(i+1) for i in range(len(varName))))
prinComp_df.to_csv('dataOUT/PrinComp.csv')

# create the correlogram of factor loadings
Rxc = pca_model.getLoadings()
# based on the ndarray of factor loadings
# vi.correlogram(matrix=Rxc, title='Correlation between the observed variables and the principal components')
# based on the DataFrame of factor loadings
Rxc_df = pd.DataFrame(data=Rxc, index=varName, columns=('C'+str(i+1) for i in range(len(varName))))
Rxc_df.to_csv('dataOUT/FactorLoadings.csv')
vi.correlogram(matrix=Rxc_df, dec=2, title='Correlation between the observed variables and the principal components')
# vi.show()

# determine the scores
scores = pca_model.getScores()
scores_df = pd.DataFrame(data=scores, index=obsName, columns=('C'+str(i+1) for i in range(len(varName))))
scores_df.to_csv('dataOUT/Scores.csv')
vi.correlogram(matrix=scores_df, dec=2, title='Correlogram of scores (standardized principle components)')
# vi.show()

# determine the quality of points representation on the axis of PC
obsQual = pca_model.getObsQual()
obsQual_df = pd.DataFrame(data=obsQual, index=obsName, columns=('C'+str(i+1) for i in range(len(varName))))
obsQual_df.to_csv('dataOUT/ObservationQuality.csv')
vi.correlogram(matrix=obsQual_df, dec=2, title='Correlogram of the quality of points representation on the axis of PC')
# vi.show()

# determine the quality of points representation on the axis of PC
betha = pca_model.getBetha()
betha_df = pd.DataFrame(data=betha, index=obsName, columns=('C'+str(i+1) for i in range(len(varName))))
betha_df.to_csv('dataOUT/Betha.csv')
vi.correlogram(matrix=betha_df, dec=2, title='Correlogram for quality of points representation on the axis of PC')
# vi.show()

# determine the commonalities (retrieving the principle components in the initial, observed variables)
commonalities = pca_model.getCommon()
commonalities_df = pd.DataFrame(data=commonalities, index=varName, columns=('C'+str(i+1) for i in range(len(varName))))
commonalities_df.to_csv('dataOUT/Commonalities.csv')
vi.correlogram(matrix=commonalities_df, dec=2, title='Correlogram commonalities (retrieving the principle components in the initial, observed variables)')
# vi.show()

# draw the correlation between the initial variable and C1, C2
vi.corrCircle(matrix=Rxc_df, title='Correlation between the initial variable and C1, C2')
# vi.show()

# in the correlation circle, draw the distribution of observations in the space of C1, C2
maxScore = np.max(scores)
minScore = np.min(scores)
print('Max. score, used as radius for the correlation circle: ', maxScore)
vi.corrCircle(matrix=scores_df, radius=maxScore, valMin=minScore, valMax=maxScore,
              title='Distribution of observations in the space of C1, C2')
vi.show()