import pandas as pd
import numpy as np
import os
import glob
import DyMMMSettings as settings

"""
Analyzes the stability of simulations by comparing the values of various state 
variables at the last and third-last time points. The function iterates over directories 
of simulation data, calculates the relative error for each state variable, and outputs 
a CSV file containing these errors for each directory.

The aim is to assess the stability of the simulation outcomes by checking how much the 
state variables change towards the end of the simulations.
"""

communitiesDir=settings.simSettings["communitiesDir"]
communityName="communitypred_cstr"
analysisDir=settings.simSettings["analysisDirName"]
dataDir=communitiesDir+'/'+communityName+'/'+analysisDir
resultsPath=communitiesDir+"/"+communityName+"/results/stabilityfiles"
if not os._exists(resultsPath):
    os.makedirs(resultsPath)

stateList=['time', 'Vol', 'biomass1', 'biomass2', 'EX_glc__D_e', 'EX_his__L_e', 'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2']

statesDictFiller={}
statesDictStateError={}
for state in stateList:
    statesDictFiller[state]=1e6
    statesDictStateError[state]=0

dirList = sorted(glob.glob(dataDir+'/*/'))

for dirName in dirList:
    stabilityErrorDf=pd.DataFrame(columns=stateList)
    fileList = glob.glob(dirName+'/*')
    print(sorted(fileList))
    for fileName in sorted(fileList):
        if (fileName[-8:]=='USER.csv'):
            continue
        if (fileName[-6:]=='_j.csv'):
            continue
        print(fileName)
        df = pd.read_csv(fileName,compression='gzip')
        #print(df)
        statesDictFiller['time']=df.iloc[-1]['time']
        if df.shape[0] < 3:
            stabilityErrorDf=stabilityErrorDf.append(statesDictFiller, ignore_index=True)
        else:
            statesDictStateError['time']=df.iloc[-1]['time']
            for stateName in stateList:
                if stateName=='time':
                    continue
                value0=df.iloc[-3][stateName]
                value1=df.iloc[-1][stateName]
                error=abs(value1-value0)/(abs(value1)+abs(value0))
                statesDictStateError[stateName]=error
            stabilityErrorDf=stabilityErrorDf.append(statesDictStateError, ignore_index=True)
    print("------------------")
    #print(stabilityErrorDf)
    outfile=resultsPath+dirName[-21:-8]+"_stability.csv"
    print(outfile)
    stabilityErrorDf.to_csv(outfile, index=False)
