'''
imports
'''
import numpy as np
import pandas as pd
import factor_analyzer as fa
import scipy.stats as stats

from utils.utils import replaceNaN, standardize
import analysis_methods.EFA_graph as EFA_graph

'''
EFA - Exploratory Factor Analysis
'''

def main():
    table = pd.read_csv('./dataIN/dataset_EFA.csv', index_col=0, na_values=':')
    obs_name = table.index.values
    var_name = table.columns.values
    values_matrix = table.values
    
    X = replaceNaN(values_matrix)
    X_std = standardize(X)
    X_std_df = pd.DataFrame(data=X_std, index=obs_name, columns=var_name)
    X_std_df.to_csv('./dataOUT/[EFA] X_std.csv')
    
    # Bartlett sphericity test
    bartlett_sphericity = fa.calculate_bartlett_sphericity(X_std_df)
    # test hypothesis
    if bartlett_sphericity[0] > bartlett_sphericity[1]:
        print('There is at least one common factor')
    else:
        print('No common factors')
        exit(-1)
        
    # Kaiser-Meyer-Olkin index
    kmo = fa.calculate_kmo(X_std_df)
    # test hypothesis
    if kmo[1] >= 0.5:
        print('There is at least one common factor')
    else:
        print('There are no common factors')
        exit(-2)

    kmo_vect = kmo[0]
    print(kmo_vect)
    # transform into matrix with 1 column
    kmo_matrix = kmo_vect[:, np.newaxis]
    print(kmo_matrix)
    kmo_df = pd.DataFrame(data=kmo_matrix, index=var_name, columns=['KMO Indices'])
    EFA_graph.correlogram(matrix=kmo_df, decimals=4, title='Correlogram of Kaiser-Meyer-Olkin indices', x_label='KMO index', y_label='variables')
    
    # ADA_EN_lecture_5.pdf - page 22, 23
    def bartlett_test(loadings, epsilon):
        # n - number of observations
        n = X.shape[0]
        m, q = np.shape(loadings)
        # print(loadings)
        R = np.corrcoef(X, rowvar=False)
        V = R
        
        psi = np.diag(epsilon)
        
        # loadings - u
        # V_estim = V cu caciula
        V_estim = loadings @ np.transpose(loadings) + psi
        
        I_estim = np.linalg.inv(V_estim) @ V
        
        determinant_I_estim = np.linalg.det(I_estim)
        
        trace_I_estim = np.trace(I_estim)
        
        if determinant_I_estim > 0:
            chi2Calc = (n - 1 - (2 * m + 4 * q - 5) / 6) * (trace_I_estim - np.log(determinant_I_estim) - m)
            # r - degrees of freedom
            df = ((m - q) * (m - q) - m - q) / 2
            chi2Tab = stats.chi2.cdf(chi2Calc, df)
            
            return chi2Calc, chi2Tab
        
        return np.nan, np.nan
    
    # significant factor extraction
    noSignificantFactors = 1
    chi2Tab_min = 1     # start at the lowest value
    # compute k iterations where k = number of variables
    for k in range(1, len(list(var_name))):
        fa_model = fa.FactorAnalyzer(n_factors=k)
        fa_model.fit(X=X_std_df)
        common_factors = fa_model.loadings_             # loadings
        specific_factors = fa_model.get_uniquenesses()  # epsilon
        
        chi2Calc, chi2Tab = bartlett_test(loadings=common_factors, epsilon=specific_factors)
        
        if np.isnan(chi2Calc) == True or np.isnan(chi2Calc) == True:
            break
        if chi2Tab_min > chi2Tab:
            noSignificantFactors = k
            chi2Tab_min = chi2Tab
            
    print('Number of significant factors ', noSignificantFactors)
    
    # create the model with the number of significant factors
    fa_fit_model = fa.FactorAnalyzer(n_factors=noSignificantFactors)
    fa_fit_model.fit(X_std_df)
    
    fa_factor_loadings = fa_fit_model.loadings_
    factor_labels = ['F' + str(i + 1) for i in range(noSignificantFactors)]
    
    fa_factor_loadings_df = pd.DataFrame(data=fa_factor_loadings, columns=factor_labels, index=var_name)
    EFA_graph.correlogram(matrix=fa_factor_loadings_df, title="Correlogram of Correlation Factors")
    
    
    # display the graph for eigenvalues
    eigenvalues = fa_fit_model.get_eigenvalues()
    print(eigenvalues)
    original_eigenvalues = eigenvalues[0]
    
    EFA_graph.principalComponents(original_eigenvalues)
            