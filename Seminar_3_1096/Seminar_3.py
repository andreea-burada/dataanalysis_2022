import numpy as np
import pandas as pd


# create a dictioanry with 4 keys
# 'S1', 'S2', ...
# and as values 5 randomly generated integers
# between 1 and 10
dict_1 = {'S'+str(j+1):
        [i for i in np.random.randint(1, 10, 5, int)]
        for j in range(4)}
print(dict_1)
for (k, v) in dict_1.items():
        print(k, ': ', v)

# df_1 = pd.DataFrame(data='garbage')
# crete a pandas DataFrame from a whole
# dictionary
df_1 = pd.DataFrame(data=dict_1)
print(df_1); print(type(df_1))

print(df_1.columns); print(type(df_1.columns))
print(list(df_1.columns))
print(type(list(df_1.columns)))

print(df_1.index); print(type(df_1.index))
print(df_1.values); print(type(df_1.values))

# create a pandas DataFrame from the values
# of a dictionary
print(type(dict_1.values()))
df_2 = pd.DataFrame(data=dict_1.values())
print(df_2)

df_3 = pd.DataFrame(data=dict_1.values(),
        columns=['V'+str(j+1) for j in range(5)],
        index=['O'+str(i+1) for i in range(4)])
print(df_3)

# create a "vector" of 100 random integers
vect_1 = np.random.randint(1, 10, 100, int)
print(vect_1); print(type(vect_1))
# the size in bytes of a vector element
print(vect_1.itemsize)

# cerate a bidimensional numpy ndarray from
# a unidimensional one
nda_1 = np.ndarray(shape=(7, 5), buffer=vect_1,
                   dtype=int, order='C')
# 'C'=Continuous, 'F'=Fortran, 'A'=Any
print(nda_1)

# using ndarray constructur with offset parameter
# the offset it's in bytes
nda_2 = np.ndarray(shape=(7, 5), buffer=vect_1,
                   offset=3*vect_1.itemsize,
                   dtype=int, order='C')
print(nda_2)

# create a pandas DataFrame from a numpy ndarray
df_4 = pd.DataFrame(data=nda_1,
    columns=['V' + str(j + 1) for j in range(5)],
    index=['O' + str(i + 1) for i in range(7)])
print(df_4)

# saving th eDataFrame into a CSV file
df_4.to_csv('Output.csv')

# upload a CSV file into a pandas DataFrame
df_5 = pd.read_csv('Output.csv',
                   index_col=0)
print(df_5)