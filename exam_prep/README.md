# Table of Contents
### [Introduction](#introduction-1)
### [PCA - Principal Component Analysis](#pca---principal-component-analysis-1)
<br></br>

---
<br></br>

# Introduction

### PDF - Probability Density Function
Measures the possiblity for a variable to take a certain value
- results are ${ \in [0,1] }$ \
$${ f(x) = P(X=x) }$$

- ${X}$ - variable
- ${x}$ - one of the values that ${X}$ may take
<br></br>
---

### CDF - Cumulative Distribution Function
Measures the possibility for continuous random variable ${X}$ to take values within a certain interval
$${ F(x) = P(X \le x ) = \int_{-\infty}^{x} f_X(t) \ dt}$$

- If ${X}$ is a purely discrete random variable, then it attains values ${x_1, x_2, ...}$ with probability ${p_i = P(x_i)}$

- **discrete random variable** = quantity whose value is determined by chance; it can only take a countable or finite number of values

- geometrically, ${F(x)}$ represents the area below the **probability density curve**
<br></br>
---
<br></br>

# PCA - Principal Component Analysis
### Definition
- `PCA` = a statistical procedure (that uses an orthogonal transformation*) that converts a set of observations of **possibly correlated variables** into a set of values of **linearly uncorrelated variabled** called `principal components`
- in other words, PCA looks to simplify the database by reducing the number of variables through grouping based on correlation
<br></br>
- `*orthogonal transformation` = is a linear transformation that preserves the inner product. That is, for each pair ${(u,v) = (Tu,Tv)}$


The first component agglutinates the most important information type because **it contains the maximum variance** \
*Geometrically*, it is all about **determining the number of axis to be chosen** for a multidimensional representation in order to obtain a satisfactory informational coverage

### Other Information
The dataset is of shape:
```math
X = 
\begin{bmatrix} 
x_{11} & ... & x_{1j} & ... & x_{1m} \\
x_{21} & ... & x_{2j} & ... & x_{2m} \\
... & ... & ... & ... & ... \\
x_{i1} & ... & x_{ij} & ... & x_{im} \\
... & ... & ... & ... & ... \\
x_{n1} & ... & x_{nj} & ... & x_{nm} \\
\end{bmatrix} 
```
, where ${x_{ij}}$ is the value taken by `variable` j for the `observation` i
- there are `m columns` -> `m variables`
- there are `n rows` -> `n observations`

- ${X_j}$ represents the column for variable ${j}$ for all ${n}$ observations

#### Keiser Criterion upon choosing number of axis
- only applicable if ${ X_j,\ j=\overline{1,m} }$ are standardized
> The Kaiser rule recommends to keep those principal components which **have a variance (eigenvalue) greater than 1**

### What PCA Does
The **goal** is to describe the table ${X}$ through a reduced number of **non-related** variables:
$${ C_1, C_2, ..., C_s }$$
To determine a new variable ${C_k}$, you need to compute the *linear combination* of variables ${X}$
$${ C_k = X \cdot a_k,\ k=\overline{1,s} }$$
, where ${s}$ is the `number of principal components`

### How To Do It
In order to apply PCA on a dataset, we will need to compute the PCA class ourselves since no external libraries are used (but they exist, we just didn't use them).

PCA analysis consists of several steps:

#### Step 1
> If data is missing, make sure to replace NaN values
```python
def replaceNaN(matrix):
    averages = np.nanmean(a=matrix, axis=0)     # axis 0 - columns
    positions = np.where(np.isnan(matrix))
    matrix[positions] = averages[positions[1]]
    return matrix
```
> Standardize the range of continuous initial variables \
> In other words, standardize ${X}$ matrix 

`standardization` = the process of putting different variables on the same scale

```python
from sklearn.preprocessing import ScandardScaler

standardScaler = StandardScaler()
X_standardized = standardScaler.fit_transform(X)    # X - matrix of values
```
OR
```python
import numpy as np

def standardize(X):
    averages = np.mean(X, axis=0)   # means on columns
    stand_dev = np.std(X, axis=0)   # standard. dev on columns
    return (X - averages) / stand_dev
```

#### Step 2
> Compute the covariance matrix to identify correlations

Using the library `numpy` we can compute the covariance matrix
```python
cov_matrix = np.cov(X_standardized, rowvar=False)
``` 
It is important to specify `rowvar=False` in order to tell the function `cov` that the variables are located on *columns* not rows.

#### Step 3
> Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components \
> Order in descending order the eigenvalues and eigenvectors and regularize the eigenvectors if necessary
```python
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```
These eigenvalues and eigenvectors need to be sorted in descending order and become `alpha` (eigenvalues) and `a` (eigenvectors)
```python
k_desc = [k for k in reversed(np.argsort(eigenvalues))]
alpha = eigenvalues[k_desc]
a = eigenvectors[:, k_desc]
```
!!! `a` needs to be regularized ~~(don't know if this word is correct in English)~~
- `regularization` = if in an eigenvector the absolute value of minimum is greater than the value of the maximum then it is statistically more useful to inverse the values of the eigenvector
```python
# we use j since it represents the column index
# eigenvectors are situated on columns, column j is the eigenvector for eigenvalue on position j from alpha
for j in range(len(alpha)):
    # current eigenvector is a[:, j]
    mini = np.min(a[:, j])
    maxi = np.max(a[:, j])
    # check if the eigenvector needs to be inverted
    if abs(mini) > abs(maxi):
        a[:, j] = -a[:, j]
```

#### Step 4
> Compute the graph in order to determine the principal components \
> As per the Kaiser rule, the eigenvalues over 1 will determine the principal components
```python
import matplotlib.pyplot as plots

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
```