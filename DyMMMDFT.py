
import sys
import math, cmath
import numpy as np
import pandas as pd
import DyMMMDataPlot
from scipy import fftpack
import matplotlib.pyplot as plt



def fft(freq, dataFrame, filePath=None):

    f_s=freq * 360

    if filePath is not None:
        dataFrame=pd.read_csv(filePath, index_col='time')

    x=dataFrame['biomass1']
    X = fftpack.fft(x.to_numpy())
    freqs = fftpack.fftfreq(len(x)) * f_s
    print(freqs)

    #magnitude = np.abs(spectrum)
    #phase = np.angle(spectrum)
    
    plt.stem(freqs, np.abs(X))    
    plt.show()
    plt.plot(np.degrees(np.angle(X)))
    plt.show()


def DFT2(freq, dataFrame, filePath=None, columnName='biomass1'):

    if filePath is not None:
        dataFrame=pd.read_csv(filePath)

    dataFrame['time']=dataFrame['time']-dataFrame['time'].iloc[0]

    Aavg=dataFrame[columnName].mean()
    Are=0
    Aim=0
    for index in range(dataFrame.shape[0]):
        Are+=(dataFrame[columnName].iloc[index]-Aavg) * math.cos(2 * np.pi * dataFrame['time'].iloc[index] * freq)
        Aim+=-1*(dataFrame[columnName].iloc[index]-Aavg) * math.sin(2 * np.pi * dataFrame['time'].iloc[index] * freq)

    count=dataFrame.shape[0]
    Are = Are/count
    Aim = Aim/count
    c = complex(Are, Aim) * 1e-4
    mag=np.absolute(c)
    phase=np.degrees(np.angle(c))
    return mag,phase


def DFT(freq, dataFrame, filePath=None, columnName='biomass1'):
    if filePath is not None:
        dataFrame=pd.read_csv(filePath)
    t = dataFrame["time"].to_numpy()
    t =t-t[0]
    y = dataFrame[columnName].to_numpy()
    z=np.exp(-1j * 2 * np.pi * freq * t)
    zcos = np.exp(1j * 2 * np.pi * freq *t)
    x=y*z
    ftime=t
    fSig1=y
    len = ftime.shape[0]

    sum1 = 0
    for m in range(1,len):
        sum1 += ((fSig1[m]+fSig1[m-1])/2)*(ftime[m]-ftime[m-1])

    DELTA = ftime[len-1] - ftime[0]
    avg1 = sum1/DELTA

    omega = 2*np.pi*freq
    re1 = []
    re2 = []
    im1 = []
    im2 = []

    for k in range(len):
        re1.append((fSig1[k]-avg1)*math.cos(omega*(ftime[k]-ftime[1])))
        im1.append(-1*(fSig1[k]-avg1)*math.sin(omega*(ftime[k]-ftime[1])))
    
    re_sum = 0
    im_sum = 0


    for m in range(1,len):
        re_sum +=  ((re1[m]+re1[m-1]))*(ftime[m]-ftime[m-1])
        im_sum += ((im1[m]+im1[m-1]))*(ftime[m]-ftime[m-1])

    realVal = re_sum/DELTA
    imagVal = im_sum/DELTA

    c = complex(realVal,imagVal) 


    mag=np.absolute(c) / 1e-4
    phase=np.degrees(np.angle(c))

    return mag,phase,c

if __name__ == '__main__':


    inFile=sys.argv[1]
    freq = int(sys.argv[2])
    #fft(1, None, outFile)
    #print(DFT2(freq, None, inFile))
    gain, phase, c = DFT(freq, None, inFile, "biomass1")
    gain2, phase2, c2 = DFT(freq, None, inFile, "biomass2")
    print(inFile+" {}, {}, {}, {}, {}, {}, {}, {}".format(str(gain), str(phase), str(c.real), str(c.imag), str(gain2), str(phase2), str(c2.real), str(c2.imag) ))

