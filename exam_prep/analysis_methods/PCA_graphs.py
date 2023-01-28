import numpy as np
import matplotlib.pyplot as plots
import seaborn

'''
Line graph for eigenvalues with horizontal line at Y = 1 in order to determine principal components
input: alpha - list of eigenvalues
'''
def eigenvaluesGraph(alpha, title="Eigenvalues - principal components variance", axisX_label="Principal Components", axisY_label="Eigenvalues"):
    # get figure
    plots.figure(title, figsize=(11, 8))
    # plot title
    plots.title(title, fontsize=18, color='k')
    # axis x label
    plots.xlabel(axisX_label, fontsize=16, color='k')
    # axis y label
    plots.ylabel(axisY_label, fontsize=16, color='k')

    alpha_labels = ['C' + str(k + 1) for k in range(len(alpha))]
    plots.plot(alpha_labels, alpha)
    plots.xticks(alpha_labels)
    # generate the red line at 1
    plots.axhline(1, color='g')

    plots.show()
    
'''
Correlogram
input: matrix - DataFrame - matrix of correlation or covariance; other DataFrames are accepted too
'''
def correlogram(matrix, decimals=2, title='Correlogram', valmin=-1, valmax=1):
    plots.figure(title, figsize=(13,15))
    plots.title(label=title, fontsize=18, color='k')
    # using seaborn library to make the heatmap
    seaborn.heatmap(data=np.round(matrix, decimals), vmin=valmin, vmax=valmax, cmap="RdBu", annot=True)
    
    plots.show()
    