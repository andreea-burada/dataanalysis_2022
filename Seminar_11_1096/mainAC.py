import pandas as pd
import utile.Utile as utl
import scipy.cluster.hierarchy as hiclu
import scipy.spatial.distance as dis
import grafice.Grafice as grf
import numpy as np


tabel = pd.read_csv('./dataIN/Indicatori.csv', index_col=0, na_values='')
# tabel = pd.read_csv('dataIN/Points-20210111.csv', index_col=0)
print(tabel)

obs = tabel.index.values
var = tabel.columns[3:].values
print(var); print(obs)

X_gross = tabel[var].values
X = utl.inlocuireNaN(X_gross)
Xstd = utl.standizare(X)
Xstd_df=pd.DataFrame(data=Xstd,
                     columns=var,
                     index=obs)

Xstd_df.to_csv('./dataOUT/Xstd.csv')

methods = list(hiclu._LINKAGE_METHODS)
print(methods); print(type(methods))

#check where a list of distances (metrics) would be available
# clustering the observation
h_1 = hiclu.linkage(y=Xstd, method = 'single', metric='euclidean')
print(h_1)

threshold, nrJonctiuni, jonctiuneDifMax= utl.calculThreshold(h_1)
print(threshold, nrJonctiuni, jonctiuneDifMax)

grf.dendrograma(h_1, etichete = obs, threshold=threshold,
                titlu='Observation clusters single-euclidean')

grf.show()

# clustering the variables

h_2 = hiclu.linkage(y=np.transpose(Xstd), method = 'single', metric='correlation')
print(h_2)

threshold, nrJonctiuni, jonctiuneDifMax= utl.calculThreshold(h_2)
print(threshold, nrJonctiuni, jonctiuneDifMax)

grf.dendrograma(h_1, etichete = var, threshold=threshold,
                titlu='Vaiables clusters single-correlation')

grf.show()

