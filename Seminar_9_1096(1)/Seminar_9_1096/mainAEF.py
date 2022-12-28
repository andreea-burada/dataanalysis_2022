import numpy as np
import pandas as pd
import utile as utl
import aef.AEF as aef
import factor_analyzer as fa
import grafice as g
from sklearn.preprocessing import StandardScaler


tabel = pd.read_csv('dataIN/MortalityEU.csv', index_col=0, na_values=':')
print(tabel)

obsNume = tabel.index.values
varNume = tabel.columns.values
matrice_numerica = tabel.values
print(tabel)

X = utl.replaceNAN(matrice_numerica)
print(X)

aefModel = aef.AEF(X)
# Xstd = aefModel.getXstd()
# A se vedea: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scalare = StandardScaler()
Xstd = scalare.fit_transform(X)

# salvarea matricei standardizate in fisier CSV
Xstd_df = pd.DataFrame(data=Xstd, index=obsNume, columns=varNume)
Xstd_df.to_csv('dataOUT/Xstd.csv')

# compute Bartlett sphericity test
sphericityBartlett = fa.calculate_bartlett_sphericity(Xstd_df)  # takes as parameter a DF with standardized values
print(sphericityBartlett)
if sphericityBartlett[0] > sphericityBartlett[1]:
    print('There is at least one common factor!')
else:
    print('There are no factors!')
    exit(-1)

# compute the Kaiser-Meyer-Olkin indices (KMO)
kmo = fa.calculate_kmo(Xstd_df)
print(kmo)
vector = kmo[0]
print(vector); print(vector.shape)
matrix = vector[:, np.newaxis]
print(matrix); print(matrix.shape)
matrix_df = pd.DataFrame(data=matrix,
            columns=['KMO_indices'],
        index=varNume)
g.corelograma(matrice=matrix_df, dec=3,
            titlu='Correlogram of Kaiser-Meyer-Olkin indices')
# g.afisare()

if kmo[1] >= 0.5:
    print('There is at least one common factor!')
else:
    print('There are no factors!')
    exit(-2)

# extract the significant factors
noOfSignificantFactors = 1
# for k in range(1, varNume.shape[0]):
for k in range(1, 4):
    faModel = fa.FactorAnalyzer(n_factors=k)
    faModel.fit(Xstd_df)
    factorLoadings = faModel.loadings_  # factor loadings
    print(factorLoadings)
    specificFactors = faModel.get_uniquenesses()  # specific factors
    print(specificFactors)

