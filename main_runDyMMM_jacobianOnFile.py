from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import DyMMMDataPlot
from numpy import linalg as LA
from scipy import linalg
from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver

stateNames=['biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']


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


def getSteadyStateValues(fileName):
    df_input=pd.read_csv(fileName,compression='gzip')
    df_input = df_input.iloc[: , 1:]
    print(df_input)
    df=pd.DataFrame(columns=df_input.columns.to_list())
    for index, row in df_input.iterrows():
        if df_input['EX_glc__D_e'][index] < 0.001:
            break
        df=df.append(row)
        ss=[]
        for index, name in enumerate(stateNames):
            ss.append(isSteadyState(df,name))
        #printStateSS(df['time'].iloc[-1], ss, stateNames)
        steadyStateReached=True
        for value in ss:
            if value==False:
                steadyStateReached=False
        if(steadyStateReached):
            break

    df.drop(['time','M1','M2'], axis=1, inplace=True)
    #print(df)
    return steadyStateReached, df.iloc[-1:]


if __name__ == '__main__':


    communitiesDir=settings.simSettings["communitiesDir"]
    communityName=settings.simSettings["communityName"]

    paramFileName=sys.argv[1]
    rowID=int(sys.argv[2])
    df_params=pd.read_csv(paramFileName)

    solverName=settings.simSettings["solverName"]
    sys.path.append(communitiesDir)
    stopTime=settings.simSettings['stopTime']

    communityDir=communitiesDir+"/"+communityName
    DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(communityName)).DyMMMCommunity
    community=DyMMMCommunity(communityName, communityDir)
 
    solver=DyMMMODESolver(community)

    eigen=np.zeros(shape=(1, len(stateNames)+1),dtype=complex)

    index=rowID       

    outFile=paramFileName+"dir\\"+communityName+"_"+'{0:05}.csv'.format(index)
    steadyStateReached, init_values= getSteadyStateValues(outFile)
    print(outFile)
    print(init_values.to_dict())
    init_values=init_values.values.flatten().tolist()

    print(df_params.iloc[index].to_dict())
    params=df_params.iloc[index]
    for paramName in df_params.columns.tolist():
        community.setParam(paramName,params[paramName])
        community.setParam('Sfeed1', 20)
        community.setParam('Fin', 0.01)

    #print("------------------")
    y=init_values.copy()
    J = np.zeros([len(y), len(y)], dtype = float)
    dy= np.zeros([len(y)], dtype = float)
    eps = 1e-2
    epsAbs = 1e-4
    #print(eps)
    for i in range(len(y)):
        y=init_values.copy()
        y[i] +=  y[i] * eps + epsAbs
        community.solve(0, y)
        dy_dict=community.calculateDerivates()
        #print(dy_dict)
        for j,name in enumerate(community.statesList):
            dy[j]=dy_dict[name]
        J[ : , i]=dy.transpose()

    # print(J)
    #w, v= LA.eig(J)
    #print("-------------")
    #print(w)

    w = linalg.eigvals(J)
    print(index)
    eigen=w

    eigenvalueFile=outFile[:-4]+'_j.csv'
    np.savetxt(eigenvalueFile, eigen.view(float))
