import pandas as pd
import numpy as np

# Function for replacing the missing values with the mean or mode
# Inlocuire celule lipsa (marcate cu NaN) cu media sau
# cu modulul (valoarea cu ce mai mare frecventa de aparitie)
def inlocuireNaN(t):
    # we continue processing only if received a pandas.DataFrame
    assert isinstance(t, pd.DataFrame), 'Only pandas.DataFrame is allowed'
    nume_coloane = list(t)  # list(t.columns)
    for v in nume_coloane:
        if any(t[v].isna()):
            if typ.is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                mode = t[v].mode()[0]
                t[v].fillna(mode, inplace=True)


def entropieShannonWheaver(t, v):
    # print(t, v, sep="\n")
    assert isinstance(t, pd.DataFrame), 'Only pandas.DataFrame is allowed!'

    x = t[v].values
    # print('x: ', x.shape[0], x.shape[1], x, sep='\n')

    sums = np.sum(x, axis=1)
    sums = 1 / sums
    # print('sums: ', sums.shape[0], sums)
    # P = np.empty(shape=(x.shape[0], x.shape[1]))
    # for j in range(P.shape[1]):
    #     P[:, j] = x[:, j] * sums
    # print('P: ', P.shape[0], P.shape[1], P)

    P_i = np.transpose(sums * np.transpose(x))
    # print('P_i: ', P_i.shape[0], P_i.shape[1], P_i)

    I = P_i > 0  # creating an index matrix associate to the strictly positive elements in P
    # print(I)
    logP = np.empty(shape=(P_i.shape[0], P_i.shape[1]))
    # print(type(logP))
    logP[I] = np.log2(P_i[I])
    # print('logP: ', logP.shape[0], logP.shape[1], logP, sep='\n')
    # PlogP = np.transpose(np.transpose(P) * np.transpose(logP))
    PlogP = P_i * logP
    # print('PlogP: ', PlogP.shape[0], PlogP.shape[1], PlogP, sep='\n')
    # Plog2P = P * np.where(P > 0, np.log2(P), P)
    # print('Plog2P: ', Plog2P, sep='\n')
    H = -np.sum(PlogP, axis=0)
    # print(H)
    return H


def disimilaritate(t, v):
    assert isinstance(t, pd.DataFrame), 'Putem procesa doar pandas.DataFrame ca parametru!'
    X = t[v].values
    sum_linii = np.sum(X, axis=1)
    R = np.transpose(sum_linii - X.T)
    Tx = np.sum(X, axis=0)  # sume pe coloane
    Tr = np.sum(R, axis=0)
    Tx[Tx == 0] = 1
    Tr[Tr == 0] = 1
    p_x = X / Tx
    p_r = R / Tr
    dif = np.abs(p_x - p_r)
    d = 0.5 * np.sum(dif, axis=0)
    s_d = pd.Series(data=d, index=v)
    return s_d