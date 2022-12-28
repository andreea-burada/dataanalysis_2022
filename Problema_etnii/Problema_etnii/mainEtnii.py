import pandas as pd
# import numpy as np
import functii as fun

# Citire populatie
tabel = pd.read_csv('./dataIN/Ethnicity.csv', index_col=0)
try:
    fun.inlocuireNaN(tabel)
except AssertionError as msg:
    print(msg)


# Citire coduri
localitati = pd.read_excel('./dataIN/CoduriRomania.xlsx', sheet_name=0, index_col=0)
judete = pd.read_excel('./dataIN/CoduriRomania.xlsx', sheet_name=1, index_col=0)
regiuni = pd.read_excel('./dataIN/CoduriRomania.xlsx', sheet_name=2, index_col=0)

print(tabel, localitati, judete, regiuni, sep='\n')

# variabile = list(tabel)[1:]
variabile = list(tabel.columns)[1:]
print(variabile)

# Calcul etnii la nivel de judet
t1 = tabel.merge(right=localitati, left_index=True, right_index=True)
print(t1)

variabile1 = variabile + ['County']
g1 = t1[variabile1].groupby(by='County').agg(func=sum)
print(g1)
g1.to_csv('./dataOUT/EthnicityJud.csv')


# Calcul etnii la nivel de regiune
t2 = g1.merge(right=judete, left_index=True, right_index=True)
variabile2 = variabile + ['Regiune']
g2 = t2[variabile2].groupby(by='Regiune').agg(func=sum)
print(g2)
g2.to_csv('./dataOUT/EthnicityReg.csv')

# Calcul etnii la nivel de macroregiune
t3 = g2.merge(right=regiuni, left_index=True, right_index=True)
variabile3 = variabile + ['MacroRegiune']
g3 = t3[variabile3].groupby(by='MacroRegiune').agg(func=sum)
print(g3)
g3.to_csv('./dataOUT/EthnicityMacroReg.csv')

# fun.entropieShannonWheaver(tabel, variabile)

# Calcul segregare la nivel de judet
entropy_jud = t1[variabile1].groupby(by='County').\
    agg(func=fun.entropieShannonWheaver, v=variabile)
print(entropy_jud)
entropy_jud.to_csv('./dataOUT/SegregareJudShannon.csv')

disim_jud = t1[variabile1].groupby(by='County').\
    agg(func=fun.disimilaritate, v=variabile)
print(disim_jud)
disim_jud.to_csv('./dataOUT/SegregareJudDisim.csv')

# Calcul indice disimilaritate la nivel de localitate
print(tabel[variabile])
disim_loc = tabel.groupby(by='City').agg(func=fun.disimilaritate, v=variabile)
print(disim_loc)
disim_loc.to_csv('./dataOUT/SegregareLocDisim.csv')