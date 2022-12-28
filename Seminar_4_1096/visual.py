import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def correlogram(R2, dec=2, title='Correlogram',
                minVal=-1, maxVal=1):
    plt.figure(title, figsize=(11, 8))
    plt.title(title, fontsize=14, color='r',
              verticalalignment='bottom')
    sb.heatmap(np.round(R2, decimals=dec), vmin=minVal, vmax=maxVal, cmap='bwr',
               annot=True)

def show():
    plt.show()