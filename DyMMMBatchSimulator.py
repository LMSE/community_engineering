import time
import sys
import os
import traceback
import pandas as pd
from pprint import pprint
import multiprocessing as mp
from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver
import time
from datetime import datetime
from DyMMMDataAnalyzer import DyMMMDataAnalyzer 
import numpy as np
from os import environ
import subprocess


class DyMMMBatchSimulator:

    columnNames=None
    CHUNKSIZE=10
    CPU_COUNT=20

    def __init__(self):
        self.communitiesDir=settings.simSettings["communitiesDir"]
        self.communityName=settings.simSettings["communityName"]
        print(self.communityName)
        sys.path.append(self.communitiesDir)
        self.communityDir=self.communitiesDir+"/"+self.communityName
        self.DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(self.communityName)).DyMMMCommunity
        self.stopTime=settings.simSettings['stopTime']
        self.solverName=settings.simSettings["solverName"]

    def process_frame(self, id, df):

        for index, row in df.iterrows():
            outFile=self.outputDir+"/"+self.communityName+"_"+'{0:05}'.format(index)
            if os.path.exists(outFile+".csv") == False:
                print([sys.executable, "main_runDyMMMOnFile.py",self.paramsFile, str(index)])
                subprocess.run([sys.executable, "main_runDyMMMOnFile.py",self.paramsFile, str(index)])
            if os.path.exists(outFile+"_j.csv") == False:
                print([sys.executable, "main_runDyMMM_jacobianOnFile2.py",self.paramsFile, str(index)])
                subprocess.run([sys.executable, "main_runDyMMM_jacobianOnFile2.py",self.paramsFile, str(index)])

            #os.system("python main_runDyMMMOnFile.py "outFile+".csv "+str(index))
        return

    def isSteadyState(self, df, colName):
        time1=df['time'].iloc[-1]
        time0=time1-1
        row0=df.loc[(df['time'] <= time0)]
        row1=df.loc[(df['time'] == time1)]  
        value0=row0[colName].iloc[-1]
        value1=row1[colName].iloc[-1]
        error=abs(value1-value0)
        return error < 1e-2

    def runSimulation(self, paramsFile):
        self.paramsFile=paramsFile
        self.outputDir=paramsFile+"dir"
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        analyzer=DyMMMDataAnalyzer(paramsFile)
        pendingList=analyzer.getPendingSimulationList()
        if len(pendingList) > 0:
            pprint(pendingList)
            print(len(pendingList))
        else:
            print("Data generation complete")
            return

        df = pd.read_csv(paramsFile)
        if (self.columnNames is None):
            self.columnNames=df.columns.tolist()        
        self.process_frame(0, df)
        return

    def runSimulationParallel(self, paramsFile):
        self.paramsFile=paramsFile
        self.outputDir=paramsFile+"dir"
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        analyzer=DyMMMDataAnalyzer(paramsFile)
        pendingList=analyzer.getPendingSimulationList()
        if len(pendingList) > 0:
            pprint(pendingList)
            print(len(pendingList))
            if(len(pendingList) > self.CPU_COUNT):
                self.CHUNKSIZE=int(len(pendingList)/self.CPU_COUNT)
            else:
                self.CHUNKSIZE=1
            print("CHUNKSIZE="+str(self.CHUNKSIZE))
        else:
            print("Data generation complete")
            return

        reader = pd.read_table(paramsFile, chunksize=self.CHUNKSIZE, sep=",")
        pool = mp.Pool(self.CPU_COUNT) # use 4 processes

        funclist = []
        id=0
        for df in reader:
                if (self.columnNames is None):
                    self.columnNames=df.columns.tolist()

                # process each data frame
                f = pool.apply_async(self.process_frame,[id, df])
                funclist.append(f)
                id+=1

        result = 0
        for f in funclist:
            try:
                id, ret = f.get(timeout=3000) # timeout in 10 seconds
            except Exception as e:
                #print("function took longer than %d seconds" % error.args[1])
                print(e)
                continue
            result+=ret
            print("frame completed------------------"+str(id))

        print("There are %d rows of data"%(result))


if __name__ == '__main__':

    paramDataFile=None
    simulationStatus=True
    index=0
    analysisDir=settings.simSettings["analysisDir"]
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
    
    simulator=DyMMMBatchSimulator()
    try:
        simulator.runSimulationParallel(paramDataFile+".csv")    
    except Exception as e:
       print(e)
       traceback.print_exc(file=sys.stdout)
       print("Exception occurred in main")


