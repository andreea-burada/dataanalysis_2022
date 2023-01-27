# Table of Contents
## [Introduction](#introduction-1)
## [PCA - Principal Component Analysis](#pca---principal-component-analysis-1)
<br></br>

---
<br></br>

# Introduction

### PDF - Probability Density Function
Measures the possiblity for a variable to take a certain value
- results are $ { \in [0,1]} $
$$ {f(x) = P(X=x)} $$

- ${X}$ - variable
- ${x}$ - one of the values that ${X}$ may take
<br></br>
---

### CDF - Cumulative Distribution Function
Measures the possibility for continuous random variable ${X}$ to take values within a certain interval
$$ { F(x) = P(X \le x ) = \int_{-\infty}^{x} f_X(t) \ dt} $$

- If ${X}$ is a purely discrete random variable, then it attains values ${x_1, x_2, ...}$ with probability ${p_i = P(x_i)}$

- **discrete random variable** = quantity whose value is determined by chance; it can only take a countable or finite number of values

- geometrically, ${F(x)}$ represents the area below the **probability density curve** (the graph with the `red line`)
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
The dataset is of shape: \
${ X = 
\begin{bmatrix} 
x_{11} & ... & x_{1j} & ... & x_{1m} \\
x_{21} & ... & x_{2j} & ... & x_{2m} \\
... & ... & ... & ... & ... \\
x_{i1} & ... & x_{ij} & ... & x_{im} \\
... & ... & ... & ... & ... \\
x_{n1} & ... & x_{nj} & ... & x_{nm} \\
\end{bmatrix} }$, where ${x_{ij}}$ is the value taken by `variable` j for the `observation` i
- there are `m columns` ${\rArr}$ `m variables`
- there are `n rows` ${\rArr}$ `n observations`

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
> Standardize the range of continuous initial variables \
> In other words, standardize ${X}$ matrix