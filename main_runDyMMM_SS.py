from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import DyMMMDataPlot


from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver

params1={
    'va1':64.10502441,
    'va2':0.418774414,
    'gammaa3':9.837324218750002,
    'gammaa4':55.45572265625,
    'gammaa5':92.2543359375,
    'gammaa6':65.32816406250001,
    'p1':37501572.265625,
    'p2':41032314.453125,
    'luxR':0.0007194824218750001,
    'lasR':0.000819091796875,
    'alpha1':6.50466796875,
    'alpha2':354.51673828125,
    'k1':68.7724609375,
    'k2':115.7021484375,
    'n':4.45703125,
    'theta':2.10126953125e-05,
    'beta':1.3278320312499995e-05
}

params={
    'va1':64.1050244140624,
    'va2':0.418774414062499,
    'gammaa1':49.375869140625,
    'gammaa2':36.732275390625,
    'gammaa3':130.507548828124,
    'gammaa4':135.697666015624,
    'gammaa5':44.0799023437499,
    'gammaa6':76.2742382812499,
    'p1':27270444.3359375,
    'p2':4641596.6796875,
    'luxR':0.0003975830078125,
    'lasR':0.000659790039062499,
    'alpha1':498.935888671875,
    'alpha2':578.029541015624,
    'k1':26.47509765625,
    'k2':23.96142578125,
    'n':3.919921875,
    'theta':0.00008245263671875,
    'beta':0.00004194384765625
}

def isSteadyState(df, colName):
    time1=df['time'].iloc[-1]
    time0=time1-1
    row0=df.loc[(df['time'] <= time0)]
    row1=df.loc[(df['time'] == time1)]  
    value0=row0[colName].iloc[-1]
    value1=row1[colName].iloc[-1]
    error=abs(value1-value0)
    return error < 1e-2


if __name__ == '__main__':


    communitiesDir=settings.simSettings["communitiesDir"]
    communityName=settings.simSettings["communityName"]

    if(len(sys.argv)>1):
        communityName=sys.argv[1]

    solverName=settings.simSettings["solverName"]
    sys.path.append(communitiesDir)
    stopTime=settings.simSettings['stopTime']
    communityDir=communitiesDir+"/"+communityName
    DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(communityName)).DyMMMCommunity
    community=DyMMMCommunity(communityName, communityDir)

    inFile="data/"+communityName
    df=pd.read_csv(inFile+".csv")
    print(df.iloc[-1])
    tStart=df['time'].iloc[-1]
    df.drop('time', axis=1, inplace=True)
    init_values=df.iloc[-1].to_numpy()
    print(init_values)
    #for paramName in params:
    #    community.setParam(paramName,params[paramName])

    solver=DyMMMODESolver(community)

    tMax=100
    
    sampleRate = 10
    stepSize=1/sampleRate
    frequency = 1
    length = 10

    amplitude=init_values[community.glucoseStateIndex]*1e-4
    t_perturb = np.linspace(0, length, sampleRate * length)  
    y_perturb = amplitude * np.sin(frequency *  2 * np.pi * t_perturb)  
    
    tEnd=tStart+stepSize
    t=None
    y=None
    perturb_index=0
    while tEnd < tMax:
        tspan = [tStart, tEnd]
        init_values[community.glucoseStateIndex]+=y_perturb[perturb_index]
        t_temp,y_temp=solver.run(tspan,'BDF', init_values)
        if t is None:
            t=t_temp
            y=y_temp
        else:
            t=np.append(t, t_temp[1:],axis = 0) 
            y=np.append(y, y_temp[1:],axis = 0) 
            print("y count rows={} time={}".format(str(y.shape), str(t[len(t)-1])))
        init_values=y[-1]
        df=pd.DataFrame(data=y,
                        index=t,
                        columns=community.statesList)
        ss1=isSteadyState(df,'biomass1')
        ss2=isSteadyState(df,'biomass2')
        print("Steady State index {} {}".format(str(ss1),str(ss2)))
        tStart+=stepSize
        tEnd+=stepSize
        perturb_index+=1
        print("y_perturb {}",str(y_perturb.shape))
        if(perturb_index >= y_perturb.shape[0]):
            break

    dataFrame=pd.DataFrame(data=y,
                    index=t,
                    columns=community.statesList)

    dataFrame.index.name = 'time'

    print(isSteadyState(dataFrame,'biomass1'))
    print(isSteadyState(dataFrame,'biomass2'))

    outFile="data/"+communityName+"_SS"
    dataFrame.to_csv(outFile+".csv", sep=',')

    community.fluxDf0.to_csv(outFile+"_HUSER.csv", sep=',', index=False)
    community.fluxDf1.to_csv(outFile+"_TUSER.csv", sep=',', index=False)


    DyMMMDataPlot.plot1(dataFrame, communityName)

    DyMMMDataPlot.plot1(None, communityName, outFile+".csv")
