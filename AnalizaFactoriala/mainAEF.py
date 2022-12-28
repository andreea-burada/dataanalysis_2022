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

X = utl.inlocuireNAN(matrice_numerica)
# print(X)

aefModel = aef.AEF(X)
Xstd = aefModel.getXstd()
# A se vedea: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# scalare = StandardScaler()
# Xstd = scalare.fit_transform(X)

# salvarea matricei standardizate in fisier CSV
Xstd_df = pd.DataFrame(data=Xstd, index=obsNume, columns=varNume)
Xstd_df.to_csv('dataOUT/Xstd.csv')

# calcul test de sfericitate Bartlett
sfericitateBartlett = fa.calculate_bartlett_sphericity(Xstd_df)  # ia ca parametru un DF cu valori standardizate
print(sfericitateBartlett)
if sfericitateBartlett[0] > sfericitateBartlett[1]:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori comuni!')
    exit(-1)

# calcul indici Kaiser-Meyer-Olkin (KMO)
kmo = fa.calculate_kmo(Xstd_df)  # ia ca parametru un DF cu valori standardizate
print(kmo)
vector = kmo[0]
print(type(vector)); print(vector.shape)
matrice = vector[:, np.newaxis]
print(matrice); print(matrice.shape)
matrice_df = pd.DataFrame(data=matrice,
        columns=['Indici_KMO'],
        index=varNume)
g.corelograma(matrice=matrice_df, dec=5,
            titlu='Corelogram indicilor Kaiser-Meyer-Olkin')
g.afisare()

if kmo[1] >= 0.5:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori comuni!')
    exit(-2)

# extragere factori semnificativi
nrFactoriSemnificativi = 1
chi2TabMin = 1
for k in range(1, varNume.shape[0]):
# for k in range(1, 4):
    faModel = fa.FactorAnalyzer(n_factors=k)
    faModel.fit(X=Xstd_df)
    factoriComuni = faModel.loadings_  # factorii comuni - factorii de corelatie
    print(factoriComuni)
    factoriSpecifici = faModel.get_uniquenesses()
    print(factoriSpecifici)

    chi2Calc, chi2Tab = aefModel.calculTestBartlett(factoriComuni, factoriSpecifici)
    print(chi2Calc, chi2Tab)
    #aefModel.calculTestBartlett(factoriComuni, factoriSpecifici)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break

    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        nrFactoriSemnificativi = k

print("Factori: ", nrFactoriSemnificativi)

# Crearea modelului cu numarul de factori semnificativi
faFitModel = fa.FactorAnalyzer(n_factors=nrFactoriSemnificativi)
faFitModel.fit(Xstd_df)

FA_factor_loadings = faFitModel.loadings_
numeFactori = ['F' + str(j + 1) for j in range(0, nrFactoriSemnificativi)]
FA_factor_loadings_df = pd.DataFrame(data=FA_factor_loadings, columns=numeFactori, index=varNume)
g.corelograma(matrice=FA_factor_loadings_df, titlu="Corelograma factorilor de corelatie din FactorAnalyzer")
#g.afisare()

# cercul corelatiilor variabilelor observate in spatiul factorilor 1 si 2
g.cerculCorelatiilor(matrice=FA_factor_loadings_df, titlu="Variabilele observate in spatiul factorilor 1 si 2")
#g.afisare()

FA_valori_proprii = faFitModel.get_eigenvalues()
print(FA_valori_proprii)

# grafic valori proprii din FactorAnalyzer
g.componentePrincipale(valoriProprii=FA_valori_proprii[0], titlu="Varianta explicata de factorii furnizati de FactorAnalyzer")
#g.afisare()

ACP_valori_proprii = aefModel.getValProp()
# grafic valori proprii din ACP
g.componentePrincipale(valoriProprii=ACP_valori_proprii, titlu="Varianta explicata de componentele principale din ACP")
#g.afisare()

# calitatea reprezentarii observatiilor pe axelor factorilor
calObs = aefModel.getCalObs()
calObs_df = pd.DataFrame(data=calObs, columns=['F' + str(j + 1) for j in range(varNume.shape[0])], index=obsNume)
g.corelograma(matrice=calObs_df, titlu="Calitatea reprezentarii observatiilor pe axelor factorilor")
g.afisare()