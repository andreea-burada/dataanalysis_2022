import numpy as np
import functions as fun
import visual as vi
import pandas as pd
import matplotlib.pyplot as plt


# create a matrix of (17, 13) of random floating-point numbers
# in the interval [1, 10]
vect_1 = fun.random(1, 10, 450)
print(vect_1); print(type(vect_1))
print(vect_1.itemsize)
matrix = np.ndarray(shape=(21, 20), buffer=vect_1,
                    dtype=float, order='C')
print(matrix)

# compute the correlation matrix
corr = np.corrcoef(x=matrix, rowvar=False)
print(corr); print(type(corr))
# vi.correlogram(corr, dec=1)
# vi.show()

# create the correlogram from a pandas DataFrame
corr_df = pd.DataFrame(data=corr,
    columns=['V'+str(j+1) for j in range(corr.shape[0])],
    index=['V'+str(i+1) for i in range(corr.shape[1])])
print(corr_df)

# vi.correlogram(corr_df, title='Correlogram from pandas DataFrame')
# vi.show()

# create the correlation circle
T = [t for t in np.arange(0, np.pi*2, 0.01)]
print(T); print(type(T))
x = [np.cos(t) for t in T]
y = [np.sin(t) for t in T]
plt.plot(x, y)
plt.show()
