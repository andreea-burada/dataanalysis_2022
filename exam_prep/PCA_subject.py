'''
imports
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
DISCLAIMER: the dataset used for this subject practice is not the same as the one given at the exam.
            The logic is the same, the code should work on any dataset with minimal modification
'''


'''
(1p) Exercise 1:
    Read the file and initialize a pandas.DataFrame with the given data
'''
data_df = pd.read_csv('./dataIN/PCA_subject.csv', index_col=0)


'''
(1p) Exercise 2:
    Print at the console the column labels, the row labels and an numpy.ndarray X containing the values
'''
column_lables = list(data_df.columns.values)
row_labels = list(data_df.index.values)
X = data_df.iloc[:, 1:].values

print('column labels', column_lables)
print('row labels', row_labels)
print('X', X)


'''
(1.5p) Exercise 3:
    Standardize X initial, write the result in the console and save it in a .csv file with the appropriate
    row and column labels
'''
def standardize(matrix):
    averages = np.mean(matrix, axis=0)
    standard_dev = np.std(matrix, axis=0)
    return (matrix - averages) / standard_dev

X_std = standardize(X)
print('X standardized', X_std)
X_std_df = pd.DataFrame(data=X_std, index=row_labels, columns=column_lables[1:])
X_std_df.to_csv('./dataOUT/[PCA_subject] X_std.csv')


'''
(1p) Exercise 4:
    In order to conduct PCA, compute and write at the console the correlation matrix R, for X standardized
    computed previously
'''
R = np.corrcoef(X_std, rowvar=False)
print('R correl. matrix', R)


'''
(1.5p) Exercise 5:
    Using the module numpy.linalg, compute and write at the console the eigenvalues and eigenvectors 
    for the R matrix
'''
eigenvalues, eigenvectors = np.linalg.eigh(R)
print('eigenvalues', eigenvalues)
print('eigenvectors', eigenvectors)


'''
(1.5) Exercise 6:
    Sort the eigenvalues and eigenvectors in descending order and write the results at the console
'''
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]
a_vect = eigenvectors[:, k_desc]
print('sorted eigenvalues', alpha)
print('sorted eigenvectors', a_vect)


'''
(1.5p) Exercise 7:
    Determine the principal components for matrix X standardized and write them at the console
'''
princ_comp = X_std @ a_vect
print('principal components', princ_comp)


'''
(1.5p) Exercise 8:
    Create a graph representing the eigenvalues in descending order and emphasize with a horizontal line
    the variance value equal to 1
'''
plt.figure('Eigenvalues graph', figsize=(14,10))
plt.title('Eigenvalues - principal components variance', fontsize=18)

plt.xlabel(xlabel='Principal Components')
plt.ylabel(ylabel='Eigenvalues')

plt.axhline(1, color='g')

component_labels = ['C' + str(i + 1) for i in range(len(alpha))]
plt.plot(component_labels, alpha)
plt.show()