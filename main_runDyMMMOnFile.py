from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import DyMMMDataPlot

from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver

stateNames=['biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']

"""
Runs a simulation of a microbial community model, checking for steady states and saving the results. The DyMMMODESolver 
is used to solve the model equations over time, and the steady state is determined by comparing the changes in state values. 
The results, including flux data and state changes, are saved to compressed CSV files for further analysis. 
"""

def printStateSS(t, ss,stateNames):
    prtStr="{},".format(str(t))
    for i,name in enumerate(stateNames):
        prtStr+="{}:{},".format(name,str(ss[i]))
    print(prtStr)

def isSteadyState(df, colName):
    time2=df['time'].iloc[-1]
    if(time2 < 3):
        return False
    time1=time2-1
    time0=time1-1
    row0=df.loc[(df['time'] <= time0)]
    row1=df.loc[(df['time'] <= time1)]  
    row2=df.loc[(df['time'] <= time2)]  
    value0=row0[colName].iloc[-1]
    value1=row1[colName].iloc[-1]
    value2=row2[colName].iloc[-1]
    currentDerivative=(value2-value1)/(time2-time1)
    prevDerivative=(value1-value0)/(time1-time0)    
    #error1=abs(currentDerivative-prevDerivative)
    error1=max(abs(currentDerivative), abs(prevDerivative))
    error2=abs(value2-value0)
    # error1=abs(currentDerivative-prevDerivative)
    # error2=abs(value2-value0)/max(value1, value0) 
    error=max(error1, error2)
    # print("----------------")
    # print(error1)
    # print(error2)    
    # print(error)
    return error < 1e-6


if __name__ == '__main__':


    communitiesDir=settings.simSettings["communitiesDir"]
    communityName=settings.simSettings["communityName"]

    paramFileName=sys.argv[1]
    paramFileIndex=int(sys.argv[2])

    outFile=paramFileName+"dir/"+communityName+"_"+'{0:05}'.format(paramFileIndex)

    solverName=settings.simSettings["solverName"]
    sys.path.append(communitiesDir)
    stopTime=settings.simSettings['stopTime']

    communityDir=communitiesDir+"/"+communityName
    DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(communityName)).DyMMMCommunity
    community=DyMMMCommunity(communityName, communityDir)

    df=pd.read_csv(paramFileName)

    print(df.iloc[paramFileIndex].to_dict())

    params=df.iloc[paramFileIndex]

    for paramName in df.columns.tolist():
        community.setParam(paramName,params[paramName])
        community.setParam('Sfeed1', 20)
        community.setParam('Fin', 0.01)

    solver=DyMMMODESolver(community)


    tStart=0
    tMax=200
    stepSize=1
    # sampleRate = 100
    # frequency = 1
    # length = 5

    # t_perturb = np.linspace(0, length, sampleRate * length)  
    # y_perturb = 1e-6 * np.sin(frequency *  q2 * np.pi * t)  

    tEnd=stepSize
    t=None
    y=None
    init_values=None
    while tEnd < tMax:
        tspan = [tStart, tEnd]
        t_temp,y_temp, status=solver.run(tspan,'BDF', init_values)
        if t is None:
            t=t_temp
            y=y_temp
        else:
            t=np.append(t, t_temp[1:],axis = 0) 
            y=np.append(y, y_temp[1:],axis = 0) 
            #print("y count {}".format(str(y.shape)))
        init_values=y[-1]
        df=pd.DataFrame(data=y,
                        index=t,
                        columns=community.statesList)
        #df['time'] = df.index.copy()    
        df.index.name = 'time'
        df.reset_index(level=0, inplace=True)
        ss=[]
        for index, name in enumerate(stateNames):
            ss.append(isSteadyState(df,name))
        printStateSS(t[-1], ss, stateNames)
        steadyStateReached=True
        for value in ss:
            if value==False:
                steadyStateReached=False
        if(steadyStateReached):
            break
        #params['Fin']=(20-df['EX_glc__D_e'].iloc[-1])/20.0
        #if params['Fin'] < 0:
        #    params['Fin'] = 0
        #community.setParam('Fin',params['Fin'])
        #print("Fin {}".format(str(params['Fin'])))
        tStart+=stepSize
        tEnd+=stepSize


    dataFrame=pd.DataFrame(data=y,
                    index=t,
                    columns=community.statesList)

    dataFrame.index.name = 'time'
    #dataFrame['time1'] = dataFrame.index.copy()
    dataFrame.reset_index(level=0, inplace=True)

    dataFrame['M1'] = dataFrame['biomass1'].diff()/dataFrame['time'].diff()
    dataFrame['M2'] = dataFrame['biomass2'].diff()/dataFrame['time'].diff()
    if 'biomass3' in dataFrame:
        dataFrame['M3'] = dataFrame['biomass2'].diff()/dataFrame['time'].diff()

    dataFrame.to_csv(outFile+".csv", sep=',', compression='gzip')

    community.fluxDf0.to_csv(outFile+"_HUSER.csv", sep=',', index=False, compression='gzip')
    community.fluxDf1.to_csv(outFile+"_TUSER.csv", sep=',', index=False, compression='gzip')

    print(isSteadyState(dataFrame,'biomass1'))
    print(isSteadyState(dataFrame,'biomass2'))

    if 'biomass3' in dataFrame:
        community.fluxDf2.to_csv(outFile+"_IUSER.csv", sep=',', index=False, compression='gzip')
        print(isSteadyState(dataFrame,'biomass3'))



    #DyMMMDataPlot.plot1(dataFrame, communityName)
    #DyMMMDataPlot.plot1(None, communityName, outFile+".csv")
