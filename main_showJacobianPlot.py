

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from cycler import cycler
from itertools import cycle
import seaborn as sns

stateNames=['biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']

index=stateNames.index('A1')

def plotComplexNumbersFromFile(eigenFile):
    eigen=np.loadtxt(eigenFile).view(complex)
    print(eigen.shape)
    X = [x[index].real for x in eigen]
    Y = [x[index].imag for x in eigen]
    print(X)
    plt.scatter(X,Y, color='red', marker=".")
    plt.title("Stability")
    plt.show()
    

if __name__ == '__main__':
    plotComplexNumbersFromFile("C:/dymmm/code/worktree/ruhycode/communities/stability/coop/high/205_j.csv")