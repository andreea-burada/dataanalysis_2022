'''
imports
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
import factor_analyzer as fa

'''
DISCLAIMER: the dataset used for this subject practice is not the same as the one given at the exam.
            The logic is the same, the code should work on any dataset with minimal modification
'''

'''
Exercise 1:
    Compute FA on the dataset and calculate the factor scores. Save the results in scores.csv
'''
dataset_df = pd.read_csv("./dataIN/EFA_subject.csv", index_col=0, na_values=':')
obs_name = dataset_df.index.values
var_name = dataset_df.columns.values
X = dataset_df.values
print(X)

def replaceNaN(matrix):
    averages = np.nanmean(matrix, axis=0)
    locations = np.where(np.isnan(matrix))
    print(locations)
    matrix[locations] = averages[locations[1]]
    
    return matrix    

X = replaceNaN(X)
#print(X)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print(X_std)
X_std_df = pd.DataFrame(data=X_std, index=obs_name, columns=var_name)

R = np.corrcoef(X_std, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(R)

k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]
a = eigenvectors[:, k_desc]

# rationalization of eigenvectors
for j in range(len(alpha)):
    mini = np.min(a[:, j])
    maxi = np.max(a[:, j])
    if abs(mini) > abs(maxi):
        a[:, j] *= -1
        
princ_comp = X_std @ a
scores = princ_comp / np.sqrt(alpha)

print(scores)
print(np.shape(scores))
scores_df = pd.DataFrame(data=scores, index=obs_name, columns=['C' + str(i + 1) for i in range(len(var_name))])
scores_df.to_csv("./dataOUT/[EFA_subject] scores.csv")

fa_model = fa.FactorAnalyzer(n_factors=np.shape(var_name)[0], rotation='varimax')
print(np.shape(var_name)[0])
fa_model.fit(X_std_df)

fa_scores = fa_model.loadings_
fa_scores_df = pd.DataFrame(data=fa_scores, index=var_name, columns=['C' + str(i + 1) for i in range(len(var_name))])
fa_scores_df.to_csv('./dataOUT/[EFA_subject] scores_fa.csv')


'''
Exercise 2:
    Draw a graph of factorial correlations between variables and factors for the first two factors
'''
corr = fa_scores[:, :2]
print(np.shape(corr))
corr_df = pd.DataFrame(data=corr, index=var_name, columns=['F' + str(i + 1) for i in range(np.shape(corr)[1])])

title = 'Correlogram of Variables and Factors'

plt.figure(title, figsize=(14,16))
plt.title(title, fontsize=18)
sb.heatmap(corr_df, vmin=-1, vmax=1, annot=True, cmap='RdBu')
plt.xlabel('Factors')
plt.ylabel('Variables')

plt.show()

