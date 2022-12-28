import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis
import Utils.Utils as utl
import Utils.Dendrograms as dnr

# Reading the input data
inFile = 'DataIN/dataset_cluster.csv'
table = pd.read_csv(inFile, index_col=0, na_values='')
obsName = table.index[:]
# print(obsName)

# Creating a ndarray which contains only the numerical data
Xraw = table.select_dtypes(include=np.number)
Xraw = table.iloc[:, 3:].values
varName = table.columns[3:]
# print(varName)
# print(len(varName))
# print(Xraw)

# Replacing the null values, if the case
Xnew = utl.replaceNaN(Xraw)

# Standardizing the values from the dataset
X = utl.standardize(Xnew)

# Printing the list of available methods for clustering
methods = list(hic._LINKAGE_METHODS)
# print(methods)

# Printing the list of distances to be used for clustering
dists = dis._METRICS_NAMES
# print(dists)

HC_1 = hic.linkage(X, method=methods[3], metric=dists[7])
a1, b1, c1 = utl.threshold(HC_1)
# print(HC_1)
# print(a1, b1, c1)
# Computing the optimum number of clusters
k1 = c1 - b1


HC_2 = hic.linkage(X, method=methods[1], metric=dists[7])
a2, b2, c2 = utl.threshold(HC_2)
# print(HC_2)
# print(a2, b2, c2)
k2 = c2 - b2

HC_3 = hic.linkage(X, method=methods[5], metric=dists[7])
a3, b3, c3 = utl.threshold(HC_3)
# print(HC_3)
# print(a3, b3, c3)
k3 = c3 - b3

# Computing the clusters belonging to the maximum stability partition --> Corresponding to the complete  method
lbl1, codes1 = utl.clusters(HC_1, k1)
lbl2, codes2 = utl.clusters(HC_2, k2)
lbl3, codes3 = utl.clusters(HC_3, k3)
# print(lbl1)
# print(codes1)

# Saving the partition of maximum stability in an output file
Cluster1 = pd.DataFrame(data=lbl1, index=obsName, columns=['Cluster'])
Cluster2 = pd.DataFrame(data=lbl2, index=obsName, columns=['Cluster'])
Cluster3 = pd.DataFrame(data=lbl3, index=obsName, columns=['Cluster'])
# print(Cluster1)
Cluster1.to_csv('DataOUT/Clusters_HC_1.csv')
Cluster2.to_csv('DataOUT/Clusters_HC_2.csv')
Cluster3.to_csv('DataOUT/Clusters_HC_3.csv')

# Clustering the observed variables
HC_4 = hic.linkage(X.transpose(), method=methods[6], metric=dists[4])

dnr.dendrogram(HC_1, labels=obsName, title='Hierarchical Classification ' + methods[3].capitalize() + ' --> '
                                           + dists[7].capitalize(), threshold=a1)
dnr.dendrogram(HC_2, labels=obsName, title='Hierarchical Classification ' + methods[1].capitalize() + ' --> '
               + dists[7].capitalize(), threshold=a2)
dnr.dendrogram(HC_3, labels=obsName, title='Hierarchical Classification ' + methods[5].capitalize() + ' --> '
               + dists[7].capitalize(), threshold=a3)
dnr.dendrogram(HC_4, labels=varName, title='Hierarchical Classification of Variables ' + methods[6].capitalize() + ' --> '
               + dists[4].capitalize())
dnr.display()
