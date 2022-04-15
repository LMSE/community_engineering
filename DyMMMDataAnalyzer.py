import time
import sys
import os
import pandas as pd
import numpy as np
from scipy import integrate
from pprint import pprint
import multiprocessing as mp
from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver

stateNames=['vol','biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']


class DyMMMDataAnalyzer:

    columnNames=None

    def __init__(self, paramsFile, communityName=None):
        self.communitiesDir=settings.simSettings["communitiesDir"]
        self.communityName=settings.simSettings["communityName"]
        if(communityName is not None):
            self.communityName=communityName
        sys.path.append(self.communitiesDir)
        self.communityDir=self.communitiesDir+"/"+self.communityName
        self.DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(self.communityName)).DyMMMCommunity
        self.stopTime=settings.simSettings['stopTime']
        self.solverName=settings.simSettings["solverName"]
        self.paramsFile=paramsFile
        self.outputDir=self.paramsFile+"dir"
        self.df=pd.read_csv(paramsFile)
        self.columnNames=self.df.columns.tolist()

    def getPendingSimulationList(self):
        pendingList=[]
        for index, row in self.df.iterrows():
            outFile=self.outputDir+"/"+self.communityName+"_"+'{0:05}'.format(index)
            if os.path.exists(outFile+".csv")==True and os.path.exists(outFile+"_j.csv")==True:
                #print("===================================================================file {} exists".format(outFile+".csv"))
                continue
            pendingList.append(outFile+".csv")
        return(pendingList)

    def appendSteadyStateStatus(self):
        None

    def appendCSI(self):
        CSIValues=[-1] * self.df.shape[0]
        self.df["CSI"]=CSIValues
        for index, row in self.df.iterrows():
            outFile=self.outputDir+"/"+self.communityName+"_"+'{0:05}'.format(index)
            simDataDF=pd.read_csv(outFile+".csv", compression='gzip')
            CSIValues[index]=self.computeCSI(simDataDF)
            #print(outFile + " "+str(CSIValues[index]))
            #print(simDataDF)
        self.df["CSI"]=CSIValues
        return self.df

    def computeCSI(self, df):
        CSI=0
        totalDiff=0
        totalBiomass=0
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
                totalDiff += abs(area_biomass1-area_biomass2)
                totalBiomass += max(area_biomass1, area_biomass2)
            CSI=1-(totalDiff/totalBiomass)
        #print(CSI)
        return CSI

    def steadyStateError(self, df, colName):
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
        return error


    def appendSS(self, colName):
        SSValues=[-1] * self.df.shape[0] #create vector of values initialized to -1
        self.df[colName+"_SS"]=SSValues
        columnValues=[-1] * self.df.shape[0] #create vector of values initialized to -1
        for index, row in self.df.iterrows():
            outFile=self.outputDir+"/"+self.communityName+"_"+'{0:05}'.format(index)
            simDataDF=pd.read_csv(outFile+".csv", compression='gzip')
            error=self.steadyStateError(simDataDF, colName)
            SSValues[index]=error
            columnValues[index]=simDataDF[colName].iloc[-1]
        self.df[colName+"_SS"]=SSValues
        self.df[colName]=columnValues
        return self.df



    def saveAnalysis(self):
        jacobianData=eigen=np.zeros(shape=(self.df.shape[0], len(stateNames)),dtype=complex)
        for index, row in self.df.iterrows():
            outFile=self.outputDir+"/"+self.communityName+"_"+'{0:05}'.format(index)
            jacobianData[index]=np.loadtxt(outFile+'_j.csv').view(complex)
        self.jacobianFile=self.paramsFile[:-4]+"_j.csv"
        print(jacobianData.shape)
        np.savetxt(self.jacobianFile, jacobianData.view(float))
        return
        self.resultFile=self.paramsFile[:-4]+"_RESULT.csv"
        self.df.to_csv(self.resultFile,sep=',',index=False)

if __name__ == '__main__':

    analysisDir=settings.simSettings["analysisDir"]
    communityName=settings.simSettings["communityName"]
    print(communityName)

    paramDataFile=None
    simulationStatus=True
    index=0
    while simulationStatus:
        inputFile = analysisDir+"/params_"+'{0:05}'.format(index)
        if os.path.exists(inputFile+".csv"):
            if os.path.exists(inputFile+"_RESULT.csv")==True and os.path.exists(inputFile+"_j.csv")==True:
                index+=1
                continue
            else:
                paramDataFile=inputFile
                break
        else:
            break

    if paramDataFile == None:
        print("nothing to simulate")
        exit(0)

    print("Checking simulation status...")
    analyzer=DyMMMDataAnalyzer(paramDataFile+".csv")
    pendingList=analyzer.getPendingSimulationList()
    if len(pendingList) > 0:
       pprint(pendingList)
       print(len(pendingList))
    else:
       print("simulation is complete") 
    df=analyzer.appendCSI()
    df=analyzer.appendSS("biomass1")
    df=analyzer.appendSS("biomass2")
    if 'biomass3' in df.columns:
        df=analyzer.appendSS("biomass3")
    print(df)
    if 'biomass3' in df.columns:
        SS_exceptions=df.loc[(df.biomass1_SS>1e-3) | (df.biomass2_SS>1e-3)|(df.biomass3_SS>1e-3)]
    else:
        SS_exceptions=df.loc[(df.biomass1_SS>1e-3) | (df.biomass2_SS>1e-3)]
    print(SS_exceptions.shape[0])
    analyzer.saveAnalysis()


