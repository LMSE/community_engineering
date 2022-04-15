
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


# def computeCSIOldold(df):
#     mu1=integrate.trapz(df['M1'],df.index)
#     mu2=integrate.trapz(df['M2'],df.index)
#     df['MAX_M1_M2'] = df[['M1','M2']].max(axis=1) - df[['M1','M2']].min(axis=1)
#     denom=integrate.trapz(df['MAX_M1_M2'],df.index)
#     CSI=1-(abs((mu1-mu2))/denom)
#     df.drop('MAX_M1_M2', axis=1, inplace=True)
#     #print('CSI={}'.format(str(CSI)))
#     return CSI

# def computeCSIold(df):
#     df.fillna(0, inplace=True)
#     df['M1_M2_DIFF'] = abs((df[['biomass1','biomass2']].max(axis=1) - df[['biomass1','biomass2']].min(axis=1)))
#     df['M1_M2_MAX']=df[['biomass1','biomass2']].max(axis=1)
#     num=integrate.trapz(df['M1_M2_DIFF'],df['time'])
#     denom=integrate.trapz(df['M1_M2_MAX'],df['time'])
#     CSI=num/denom
#     if(CSI < 0 or CSI > 1 or math.isnan(CSI)):
#         print(df)
#         print("Error "+str(CSI)+ " denom "+str(denom))

#     df.drop('M1_M2_DIFF', axis=1, inplace=True)
#     df.drop('M1_M2_MAX', axis=1, inplace=True)
#     return CSI

# def computeCSI(df):
#     CSI=0
#     integral_biomass1=0
#     integral_biomass2=0
#     for row in range(1,df.shape[0]):
#         delta=df.iloc[row]['time']-df.iloc[row-1]['time']
#         area_biomass1=abs(df.iloc[row]['biomass1']-df.iloc[row-1]['biomass1'])*delta
#         area_biomass2=abs(df.iloc[row]['biomass2']-df.iloc[row-1]['biomass2'])*delta
#         integral_biomass1 += area_biomass1
#         integral_biomass2 += area_biomass2
#     CSI=1-(abs(integral_biomass1-integral_biomass2)/(integral_biomass1+integral_biomass2))
#     #print(CSI)
#     return CSI

def computeCSI(df):
    CSI=0
    integral_biomass1=0
    integral_biomass2=0
    lastBiomass1=df['biomass1'].iloc[-1]
    lastBiomass2=df['biomass2'].iloc[-1]
    #maxBiomass1=df['biomass1'].max()
    #maxBiomass2=df['biomass2'].max()
    if (lastBiomass1 > 1e-3 or lastBiomass2 > 1e-3):
        for row in range(1,df.shape[0]):
            delta=df.iloc[row]['time']-df.iloc[row-1]['time']
            area_biomass1=abs(df.iloc[row]['biomass1']-df.iloc[row-1]['biomass1'])*delta
            area_biomass2=abs(df.iloc[row]['biomass2']-df.iloc[row-1]['biomass2'])*delta
            integral_biomass1 += area_biomass1
            integral_biomass2 += area_biomass2
        CSI=1-(abs(integral_biomass1-integral_biomass2)/(integral_biomass1+integral_biomass2))
    #print(CSI)
    return CSI

def calculatResult(param_df, dataFilePath, df, paramFile, index):
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

    communityName="communitypred_cstr"
    processCommunity(communitiesDir, communityName, analysisDir)

