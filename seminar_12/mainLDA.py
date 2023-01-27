import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as disc
from utile import utile as utl
from grafice import grafice as graf

# LDA - Linear Discriminant Analysis

f1 = 'dataIN/ProiectB.csv'
f2 = 'dataIN/ProiectBEstimare.csv'
t1 = pd.read_csv(f1, index_col=0)
t2 = pd.read_csv(f2, index_col=0)

print(t1); print(t2)

# convert the categorical variables to quantitative variables
var_categorical = t1.columns[:6].values
print(var_categorical)

utl.coding(t1, var_categorical)
utl.coding(t2, var_categorical)
print(t1); print(t2)

#
var_predictor = t1.columns[:11].values
var_classification = "VULNERAB"
X = t1[var_predictor].values
Y = t1[var_classification].values
print(Y)

# create a linear discriminant analysis object
objectLDA = disc.LinearDiscriminantAnalysis()
objectLDA.fit(X=X, y=Y)

# compute the accuracy of prediction on the base dataset
classSetBase = objectLDA.predict(X=X) # list of values
print(classSetBase)
# we compare the Y provided with Y' (classSetBase)
classSetBase_df = pd.DataFrame(data={str(var_classification): Y, 'Prediction' + (str(var_classification)) : classSetBase}, index=t1.index.values)
classSetBase_df.to_csv('./dataOUT/PredictionSetBase.csv')

# extract rows with different predictions
errorClassSetBase_df = classSetBase_df[Y != classSetBase]
print(errorClassSetBase_df)
# compute global level of prediction accuracy -> (n - n_err) / n * 100
n = len(Y)
n_error = len(errorClassSetBase_df.index)
print(n, n_error)

accuracy_level = (n - n_error) * 100 / n
print(f"The global level of accuracy is {accuracy_level}")

# prediction on the test dataset
classSetTest = objectLDA.predict(t2[var_predictor].values)
classSetTest_df = pd.DataFrame(data={"Prediction": classSetTest}, index=t2.index.values)
classSetTest_df.to_csv('./dataOUT/PredictionSetTest.csv')

# compute the group accuracy
g = objectLDA.classes_
print(g)

q = len(g)

# compute matrix of groups where the principal diagonal shows the no. of correct prediction
groups_matrix_df = pd.DataFrame(data=np.zeros(shape=(q, q)), columns=g, index=g)
#                                                                                                                                                              print(groups_matrix_df)

for i in range(n):
    groups_matrix_df.loc[Y[i], classSetBase[i]] += 1
print(groups_matrix_df)

group_accuracy = np.diag(groups_matrix_df) * 100 / np.sum(groups_matrix_df, axis=1)
print(f"Accuracy by group: \n{group_accuracy}"); print(type(group_accuracy))
groups_matrix_df['Group Accuracy'] = group_accuracy
groups_matrix_df.to_csv('./dataOUT/GroupAccuracy.csv')


