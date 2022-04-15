from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import DyMMMDataPlot


from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver


def isSteadyState(df, colName):
    time1=df['time'].iloc[-1]
    time0=time1-1
    row0=df.loc[(df['time'] <= time0)]
    row1=df.loc[(df['time'] == time1)]  
    value0=row0[colName].iloc[-1]
    value1=row1[colName].iloc[-1]
    error=abs(value1-value0)
    return error < 1e-2

def run(init_values, tStart, freq, samplingFreq, cycles, text):

    solverName=settings.simSettings["solverName"]

    amplitude=init_values[community.glucoseStateIndex]*1e-2
    stepSize=1/samplingFreq
    stabilitySteps=samplingFreq*cycles
    tEnd=tStart+stepSize
    t=None
    y=None
    for index in range(0,stabilitySteps):
        tspan = [tStart, tEnd]
        y_perturb = amplitude * np.sin(freq *  2 * np.pi * index * stepSize)
        init_values[community.glucoseStateIndex]+=y_perturb
        t_temp,y_temp, status=solver.run(tspan,solverName, init_values)
        if t is None:
            t=t_temp
            y=y_temp
        else:
            t=np.append(t, t_temp[1:],axis = 0) 
            y=np.append(y, y_temp[1:],axis = 0) 
            print(text+"y count rows={} time={}".format(str(y.shape), str(t[len(t)-1])))
        init_values=y[-1]
        tStart+=stepSize
        tEnd+=stepSize
    return t, y


if __name__ == '__main__':

    inFile=sys.argv[1]
    freq = int(sys.argv[2])
    outFile=inFile[:-4]+"_{0:05}_FREQ.csv".format(freq)

    if os.path.exists(outFile):
        print("File exists "+outFile)
        exit(0)


    communitiesDir=settings.simSettings["communitiesDir"]
    communityName=settings.simSettings["communityName"]

    sys.path.append(communitiesDir)
    stopTime=settings.simSettings['stopTime']
    communityDir=communitiesDir+"/"+communityName
    DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(communityName)).DyMMMCommunity
    community=DyMMMCommunity(communityName, communityDir)

    df=pd.read_csv(inFile)
    print(df.iloc[-1])
    tStart=df['time'].iloc[-1]
    df.drop('time', axis=1, inplace=True)
    init_values=df.iloc[-1].to_numpy()
    print(init_values)
    #for paramName in params:
    #    community.setParam(paramName,params[paramName])


    solver=DyMMMODESolver(community)

    samplingFreq = freq * 10

    stabilityCycles=5

    t,y = run(init_values, tStart, freq, samplingFreq, stabilityCycles,"Stability--->")

    print(y)

    init_values=y[-1]
    tStart=t[-1]


    samplingFreq = freq * 360

    t,y = run(init_values, tStart, freq, samplingFreq, 1,"Sampling--->")


    dataFrame=pd.DataFrame(data=y,
                    index=t,
                    columns=community.statesList)

    dataFrame.index.name = 'time'

    print(dataFrame)

    print(isSteadyState(dataFrame,'biomass1'))
    print(isSteadyState(dataFrame,'biomass2'))
    dataFrame.to_csv(outFile, sep=',')

