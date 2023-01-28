import pandas
import sklearn.cross_decomposition as skl
import pandas as pd
import numpy as np

import utils
import utils as utl
import visual as vi

# Import 2 data sets from a single csv file
input_file = "./dataIN/Energy.csv"
table = pd.read_csv(input_file, index_col=0)

# TODO
# max perechi canonice = min (p, q)
print(table)

var_name = table.columns[1:].values
print(var_name)
obs_name = table.index.values
print(obs_name)

x_var = var_name[:4]
print(x_var)
y_var = var_name[4:]
print(y_var)

X = table[x_var].values
print(X)
X_std = utils.standardize(X)
X_std_df = pandas.DataFrame(data=X_std, index=obs_name, columns=x_var)
X_std_df.to_csv("./dataOUT/X_standardizat.csv")

Y = table[y_var].values
print(Y)
Y_std = utils.standardize(Y)
Y_std_df = pandas.DataFrame(data=Y_std, index=obs_name, columns=y_var)
Y_std_df.to_csv("./dataOUT/Y_standardizat.csv")

n, p = np.shape(X)
q = np.shape(Y)[1]

print("n, p, q: ", n, p, q)

# number of canonical pairs
m = min(p, q)
CCA_model = skl.CCA(n_components=m)
CCA_model.fit(X=X_std, Y=Y_std)

# z = CCA_model.x_values_   # depreciated
# print(z)

z, u = CCA_model.transform(X=X_std, Y=Y_std)
print(z)
print(u)

z = np.fliplr(z)    # the variance depreciates in reverse, we need to flip the columns
z_df = pandas.DataFrame(data=z, index=obs_name, columns=['z' + str(i + 1) for i in range(p)])
z_df.to_csv("./dataOUT/z.csv")
vi.corelograma(matrice=z_df, titlu="Variabilele canonice z strandadizate")

u = np.fliplr(u)    # the variance depreciates in reverse, we need to flip the columns
u_df = pandas.DataFrame(data=u, index=obs_name, columns=['u' + str(i + 1) for i in range(q)])
u_df.to_csv("./dataOUT/u.csv")
vi.corelograma(matrice=u_df, titlu="Variabilele canonice u strandadizate")

#vi.afisare()

# extragere factori de corelatie (factor loadings)
# corelatia dintre variabilele cauzale X si variabilele canonice z
R_Xz = CCA_model.x_loadings_
print(R_Xz)
R_Xz_df = pandas.DataFrame(data=R_Xz, index=[x_var], columns=['z' + str(i + 1) for i in range(p)])
vi.corelograma(matrice=R_Xz_df, titlu="Corelatia dintre X si z")

# corelatia dintre variabilele cauzale Y si variabilele canonice u
R_Yu = CCA_model.y_loadings_
print(R_Yu)
R_Yu_df = pandas.DataFrame(data=R_Yu, index=[y_var], columns=['u' + str(i + 1) for i in range(q)])
vi.corelograma(matrice=R_Yu_df, titlu="Corelatia dintre Y si u")

vi.afisare()
