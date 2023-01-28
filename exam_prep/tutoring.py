import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.cross_decomposition as skl
import matplotlib.pyplot as plots

table_industry = pd.read_csv(filepath_or_buffer="./dataIN/tutoring_Industry.csv", index_col=0)
table_populations = pd.read_csv(filepath_or_buffer="./dataIN/tutoring_Localities.csv", index_col=0)

industries_list = list(table_industry.columns[1:].values)


# exercise 1
table_1 = table_industry.merge(right=table_populations, left_index=True, right_index=True)
print(table_1)

def cifra_afaceri_loc(table, variables, population):
    x = table[variables].values / table[population]
    v = list(x)
    v.insert(0, table["Localitate_x"])
    return pd.Series(data=v, index=['Localitate'] + variables)

table_2 = table_1[['Localitate_x', 'Populatie'] + industries_list].apply(func=cifra_afaceri_loc, axis=1, variables=industries_list, population='Populatie')
print(table_2)


# exercise 2
table_3 = table_1[industries_list + ['Judet']].groupby(by='Judet').agg(sum)
print(table_3)

def max_cifra_afaceri(table):
    x = table.values    # matrix of values
    max_line = np.argmax(x)
    print(max_line)
    return pd.Series(data=[table.index[max_line], x[max_line]], index=['Activitate', 'CifraAfaceri'])

table_4 = table_3[industries_list].apply(func=max_cifra_afaceri, axis=1)
print(table_4, type(table_4))
# table_4.to_csv("./dataOUT/")


# exercise 3
canon_table = pd.read_csv("dataIN/DataSet_34.csv", index_col=0)

obs_name = canon_table.index.values
var_name = canon_table.columns.values

set_x = var_name[:4]
set_y = var_name[4:]
print(set_x, set_y)

X_matrix = canon_table[set_x].values
Y_matrix = canon_table[set_y].values

# standardize matrixes
X_df = pd.DataFrame(data=X_matrix, index=obs_name, columns=set_x)
Y_df = pd.DataFrame(data=Y_matrix, index=obs_name, columns=set_y)

scaler = StandardScaler()

X_std = scaler.fit_transform(X_matrix)
Y_std = scaler.fit_transform(Y_matrix)

X_std_df = pd.DataFrame(data=X_std, index=obs_name, columns=set_x)
Y_std_df = pd.DataFrame(data=Y_std, index=obs_name, columns=set_y)

print(X_std_df)
print(Y_std_df)

X_std_df.to_csv("./dataOUT/X_std.csv")
Y_std_df.to_csv("./dataOUT/Y_std.csv")


# exercise 4
'''
CCA - Cross Canonical Analysis

n - number of observations
p - number of variables in set X
q - number of variables in set Y
m - the minimum between p and q
'''
n, p = np.shape(X_matrix)
q = np.shape(Y_matrix)[-1]
m = min(p, q)

model_CCA = skl.CCA(m)
# using matrixes not DataFrames
model_CCA.fit(X=X_std, Y=Y_std)
# z = model_CCA.x_scores_
z, u = model_CCA.transform(X=X_std, Y=Y_std)

z_df = pd.DataFrame(data=z, index=obs_name, columns=['Z' + str(i + 1) for i in range(p)])
z_df.to_csv("./dataOUT/z_scores.csv")
u_df = pd.DataFrame(data=u, index=obs_name, columns=['U' + str(i + 1) for i in range(q)])
u_df.to_csv("./dataOUT/u_scores.csv")


# exercise 5
R_xz = model_CCA.x_loadings_
R_xz_df = pd.DataFrame(data=R_xz, index=set_x, columns=['Z' + str(i + 1) for i in range(p)])

R_yu = model_CCA.y_loadings_
R_yu_df = pd.DataFrame(data=R_yu, index=set_y, columns=['U' + str(i + 1) for i in range(q)])


# exercise 6
def biplot(x, y, xLabel='X', yLabel='Y', title='Biplot', labels_dots=None):
    figure = plots.figure(figsize=(10, 7))
    ax = figure.add_subplot(1, 1, 1)
    assert isinstance(ax, plots.Axes)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    
    ax.scatter(x=x[:, 0], y=y[:, 0], c='r', label='Set X')
    ax.scatter(x=x[:, 1], y=y[:, 1], c='b', label='Set Y')
    
    if labels_dots is not None:
        for i in range(len(labels_dots)):
            # draw for both scatters
            # z1, u1 space
            ax.text(x=x[i, 0], y=y[i, 0], s=labels_dots[i])
            # z2, u2 space
            ax.text(x=x[i, 1], y=y[i, 1], s=labels_dots[i])
    
    ax.legend()
    
biplot(z[:, :2], u[:, :2], xLabel='(z1, u1)', yLabel='(z2, u2)', labels_dots = obs_name)
plots.show()