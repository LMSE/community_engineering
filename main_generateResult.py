
import time
import sys
import os
import pandas as pd
from scipy import integrate
from pprint import pprint
import multiprocessing as mp
from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver



def computeCSI(self, df):
    """
    Computes the Community Stability Index (CSI) based on biomass data.

    Parameters:
        df (DataFrame): A DataFrame containing biomass data for the community.

    Returns:
        float: The computed CSI value.
    """
    CSI=0
    lastBiomass1=df['biomass1'].iloc[-1]
    lastBiomass2=df['biomass2'].iloc[-1]
    if (lastBiomass1 > 1e-3 or lastBiomass2 > 1e-3):
        p1=lastBiomass1/(lastBiomass1+lastBiomass2)
        p2=lastBiomass2/(lastBiomass1+lastBiomass2)
        Sp1=p1*np.log(p1)
        Sp2= p2*np.log(p2)
        CSI=  (Sp1+Sp2)/(np.log(2)) * (-1)
    #print(CSI)
    return CSI

def calculatResult(param_df, dataFilePath, df, paramFile, index):
    """
    Calculates the result for a single parameter set index and appends the CSI.

    Parameters:
        param_df (DataFrame): DataFrame containing the parameters.
        dataFilePath (str): Path to the data file.
        df (DataFrame): DataFrame containing the result data.
        paramFile (str): Path to the parameter file.
        index (int): Index of the parameter set being processed.

    Returns:
        list: A list containing the last row of the result DataFrame and the CSI value.
    """
    CSI=computeCSI(df)
    #print("{},{}".format(dataFilePath,str(CSI)))
    biomass1=df.iloc[-1]['biomass1']
    biomass2=df.iloc[-1]['biomass2']
    if  biomass2 < 0 or  biomass1 < 0:
        print("python main_runDyMMMOnFile {} {}".format(paramFile, str(index)))
        #print("{},{},{}".format(str(biomass1),str(biomass2),dataFilePath))
    #print(df.iloc[-1].tolist())
    result = df.iloc[-1].tolist().copy()
    result.append(CSI)
    return result

def processCommunity(communitiesDir, communityName, analysisDir):
    index=0
    resultDF=None
    createDF=True
    resultIndex=0
    paramDF=None
    print(communityName)
    while 1:
        inputFile = communitiesDir+"/"+communityName+"/"+analysisDir+"/params_"+'{0:05}'.format(index)
        paramFile = inputFile+".csv"
        #print(paramFile)
        if os.path.exists(paramFile):
            param_df=pd.read_csv(paramFile)
            resultDir = inputFile+".csvdir"
            if os.path.exists(resultDir):
                #print(resultDir)
                innerIndex=0
                while 1:
                    dataFile = resultDir+"/"+communityName+"_"+'{0:05}.csv'.format(innerIndex)
                    #print(dataFile)
                    if os.path.exists(dataFile):
                        #print(dataFile)
                        df=pd.read_csv(dataFile, compression='gzip')
                        if createDF:
                            paramDF=pd.DataFrame(data=None, columns=param_df.columns)
                            resultColumns=df.columns.tolist().copy()
                            resultColumns.append('CSI')
                            resultDF=pd.DataFrame(data=None, columns=resultColumns)                          
                            createDF=False
                        y=calculatResult(param_df, dataFile, df, paramFile, innerIndex)
                        paramDF.loc[resultIndex]=param_df.iloc[innerIndex]
                        resultDF.loc[resultIndex] = y

                    else:
                        break
                    innerIndex+=1
                    resultIndex+=1
        else:
            break
        index+=1
    print(paramDF)
    print(resultDF)
    outputFile=communitiesDir+"/"+communityName+"_param.csv"
    paramDF.to_csv(outputFile,  index=False)
    outputFile=communitiesDir+"/"+communityName+"_RESULT.csv"
    resultDF.to_csv(outputFile, index=False)

if __name__ == '__main__':

    communitiesDir=settings.simSettings["communitiesDir"]
    analysisDir=settings.simSettings["analysisDirName"]

    communityName="communitycoop_cstr"
    processCommunity(communitiesDir, communityName, analysisDir)

    communityName="communitycomp_cstr"
    processCommunity(communitiesDir, communityName, analysisDir)

    communityName=settings.simSettings["communityName"]
    processCommunity(communitiesDir, communityName, analysisDir)

