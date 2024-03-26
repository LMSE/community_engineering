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

#stateNames=['vol','biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']
stateNames=[]

class DyMMMDataAnalyzer:
    """
    A class for analyzing data generated from DyMMM simulations. It provides functionalities to determine
    the pending simulations, compute community similarity indices (CSI), and check for steady states.

    Attributes:
        columnNames (list): List of column names in the data frame.
        communitiesDir (str): Directory containing community modules.
        communityName (str): Name of the specific community module to be used.
        communityDir (str): Directory of the specific community module.
        DyMMMCommunity (module): The DyMMMCommunity module imported dynamically based on communityName.
        stopTime (int): Time to stop the simulation.
        solverName (str): Name of the solver to be used.
        paramsFile (str): Path to the parameters file.
        outputDir (str): Directory where output files are saved.
        df (DataFrame): Data frame containing parameters and results of simulations.
    """
    columnNames=None

    def __init__(self, paramsFile, communityName=None):
        """
        Initializes the DyMMMDataAnalyzer class by setting up directories, importing community modules,
        and loading the simulation parameters data frame.

        Args:
            paramsFile (str): Path to the file containing parameters for the simulation.
            communityName (str, optional): Name of the community module to be used. If not specified,
                                           the default community name from the settings is used.
        """
        self.communitiesDir=settings.simSettings["communitiesDir"]
        self.communityName=settings.simSettings["communityName"]
        if(communityName is not None):
            self.communityName=communityName
        sys.path.append(self.communitiesDir)
        self.communityDir=self.communitiesDir+"/"+self.communityName
        self.DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(self.communityName)).DyMMMCommunity
        global stateNames
        stateNames=self.DyMMMCommunity.statesList
        self.stopTime=settings.simSettings['stopTime']
        self.solverName=settings.simSettings["solverName"]
        self.paramsFile=paramsFile
        self.outputDir=self.paramsFile+"dir"
        self.df=pd.read_csv(paramsFile)
        self.columnNames=self.df.columns.tolist()

    def getPendingSimulationList(self):
        """
        Identifies the simulations that are pending based on the output files' existence.

        Returns:
            list: A list of output files that are pending for simulation.
        """
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
        """
        Appends the Community Similarity Index (CSI) values to the data frame for each simulation.

        Returns:
            DataFrame: The updated data frame with CSI values appended.
        """
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
        """
        Computes the Community Similarity Index (CSI) based on the relative abundances of biomass in the community.

        Args:
            df (DataFrame): The data frame containing simulation results.

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

    def steadyStateError(self, df, colName):
        """
        Determines the error in reaching steady state
        """ 
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
        """
        Appends steady-state error values and the last value of the specified column to the data frame. 
        This method calculates the steady-state error for each simulation result and updates the data frame.

        Args:
            colName (str): The column name for which the steady-state analysis will be performed.

        Returns:
            DataFrame: The updated data frame with steady-state error and the last value of the specified column appended.
        """
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
        """
        Saves the analysis results to files. This includes saving the jacobian data and the updated data frame
        with all analyses (steady state, CSI, etc.). The jacobian data is saved to a '_j.csv' file, and the
        data frame is saved to a '_RESULT.csv' file.

        The method iterates through each simulation output, loads the jacobian data, and aggregates it before saving.
        """
        jacobianData=eigen=np.zeros(shape=(self.df.shape[0], len(stateNames)),dtype=complex)
        for index, row in self.df.iterrows():
            outFile=self.outputDir+"/"+self.communityName+"_"+'{0:05}'.format(index)
            jacobianData[index]=np.loadtxt(outFile+'_j.csv').view(complex)
        self.jacobianFile=self.paramsFile[:-4]+"_j.csv"
        print(jacobianData.shape)
        np.savetxt(self.jacobianFile, jacobianData.view(float))
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


