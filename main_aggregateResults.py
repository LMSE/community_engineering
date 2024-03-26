
import os
import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import DyMMMSettings as settings

"""
This code aggregates and processes data from a series of files related to a biological community analysis, consolidating parameters, 
results, eigenvalues, and stability data into combined files for further analysis. The code iteratively reads multiple data files, appends 
their contents to aggregate DataFrames or arrays, and then saves the consolidated data to new files for each data type.
"""

stateNames=['biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']

communitiesDir=settings.simSettings["communitiesDir"]
communityName="communitypred_cstr"
analysisDir=settings.simSettings["analysisDirName"]

dataDir=communitiesDir+"/"+communityName+'/'+analysisDir
resultsDir=communitiesDir+"/"+communityName+'/'+"results"

cwd = os.getcwd()
os.chdir(dataDir)
result = sorted(glob.glob('*_j.csv'))
fileList=[dataDir+'/'+s for s in result]
os.chdir(cwd)

print(len(result))

initialized=False

eigenData=None
paramDF=None
resultDF=None

i=0
for file in fileList:
    i=i+1
    jacobianFile=file
    paramFile=file[:-6]+'.csv'
    resultFile=file[:-6]+'_RESULT.csv'
    stabilityFile=file[:-6]+'_stability.csv'

    #print(jacobianFile, paramFile, resultFile)
    eigenDataIn=np.loadtxt(jacobianFile).view(complex)
    paramDFIn=pd.read_csv(paramFile)
    resultDFIn=pd.read_csv(resultFile)
    stabilityDFIn=pd.read_csv(stabilityFile)
    if initialized:
        paramDF=paramDF.append(paramDFIn, ignore_index=True)
        resultDF=resultDF.append(resultDFIn, ignore_index=True)
        stabilityDF=stabilityDF.append(stabilityDFIn, ignore_index=True)
        if len(eigenDataIn.shape) == 1:
            eigenDataIn=eigenDataIn.reshape(-1, eigenData.shape[1])
        eigenData=np.append(eigenData, eigenDataIn,axis = 0)
    else:
        paramDF=paramDFIn
        resultDF=resultDFIn
        eigenData=eigenDataIn
        stabilityDF=stabilityDFIn
        initialized=True

paramDF.to_csv(resultsDir+'/'+communityName+'_params.csv', index=False)
resultDF.to_csv(resultsDir+'/'+communityName+'_results.csv', index=False)
np.savetxt(resultsDir+'/'+communityName+'_eigen.npy', eigenData.view(float))
stabilityDF.to_csv(resultsDir+'/'+communityName+'_stability.csv', index=False)

