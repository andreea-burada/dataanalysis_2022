'''
imports
'''
import numpy as np
import pandas as pd
import sklearn.cross_decomposition as skl
import matplotlib.pyplot as plt

from utils.utils import standardize

def main():
    table = pd.read_csv("./dataIN/dataset_CCA.csv", index_col=0)
    print(table)
    var_name = table.columns.values
    obs_name = table.index.values
    country_names = obs_name
    
    # split dataset into 2 sets
    # X - first 4 variables
    # Y - last 4 variables
    x_var = var_name[:4]
    y_var = var_name[4:]
    
    print(x_var, y_var)
    
    X = table[x_var].values
    Y = table[y_var].values
    
    print(X, Y)
    
    X_std = standardize(X)
    Y_std = standardize(Y)
    
    X_std_df = pd.DataFrame(data=X_std, index=obs_name, columns=x_var)
    Y_std_df = pd.DataFrame(data=Y_std, index=obs_name, columns=y_var)
    
    X_std_df.to_csv("./dataOUT/[CCA] X_std.csv")
    Y_std_df.to_csv("./dataOUT/[CCA] Y_std.csv")
    
    # n - rows; no. observations
    n = np.shape(obs_name)[0]
    
    # p, q
    p = np.shape(x_var)[0]
    q = np.shape(y_var)[0]
    
    # m - ADA_EN_lecture_9.pdf - page 22
    m = min(p, q)
    
    CCA_model = skl.CCA(n_components=m)
    CCA_model.fit(X=X_std, Y=Y_std)
    
    # get z, u
    z, u = CCA_model.transform(X=X_std, Y=Y_std)
    
    # IMPORTANT!!!!!!
    z = np.fliplr(z)
    u = np.fliplr(u)
    
    # z - linear combination
    z_df = pd.DataFrame(data=z, index=obs_name, columns=['Z' + str(i + 1) for i in range(p)])
    z_df.to_csv("./dataOUT/[CCA] z_scores.csv")
    
    u_df = pd.DataFrame(data=u, index=obs_name, columns=['U' + str(i + 1) for i in range(q)])
    u_df.to_csv("./dataOUT/[CCA] u_scores.csv")
    
    # correlation matrices
    # R_Xz
    R_Xz = CCA_model.x_loadings_
    R_Xz_df = pd.DataFrame(data=R_Xz, index=[x_var], columns=['z' + str(i + 1) for i in range(p)])
    
    # R_Yu
    R_Yu = CCA_model.y_loadings_
    R_Yu_df = pd.DataFrame(data=R_Yu, index=[y_var], columns=['u' + str(i + 1) for i in range(p)])
    
    # scatter plot in space (z1, u1) (z2, u2)
    z_space = z[:, :2]
    u_space = u[:, :2]
    
    title = 'Scatter in (z1, u1); (z2, u2)'
    figure = plt.figure(title, figsize=(14,15))
    subplot = figure.add_subplot()
    
    plt.title(title, fontsize=16)
    
    plt.xlabel('z1, z2')
    subplot.scatter(x=z_space[:, 0], y=u_space[:, 0], c='b', label='Set X')
    
    plt.ylabel('u1, u2')
    subplot.scatter(x=z_space[:, 1], y=u_space[:, 1], c='g', label='Set Y')
    subplot.legend()
    
    for i in range(np.shape(country_names)[0]):
        subplot.text(x=z_space[i, 0], y=u_space[i, 0], fontsize=9, s=country_names[i])
        subplot.text(x=z_space[i, 1], y=u_space[i, 1], fontsize=9, s=country_names[i])
    
    plt.show()