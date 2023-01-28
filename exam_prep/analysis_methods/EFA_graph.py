'''
imports
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def correlogram(matrix: pd.DataFrame, title='Correlogram', decimals=2, min_val=-1, max_val=1, x_label='X', y_label='Y'):
    plt.figure(title, figsize=(10, 13))
    plt.title(title, fontsize=18, color='g')
    sb.heatmap(data=np.round(matrix, decimals), vmin=min_val, vmax=max_val, cmap='RdBu', annot=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.show()
    
def principalComponents(alpha: np.ndarray, title='Variance Explained by Principal Components'):
    plt.figure(title, figsize=(14,14))
    # get labels
    alpha_labels = ['C' + str(i + 1) for i in range(len(alpha))]
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    plt.axhline(1, color='r')
    plt.plot(alpha_labels, alpha, 'go-')
    
    plt.show()