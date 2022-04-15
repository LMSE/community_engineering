import os
import sys

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
import DyMMMSettings as settings

"""The code generates the space filing samples in the specified directory."""
"""copy or create the screening_inputparams.csv"""

class DyMMMSamplesSobol:

    problemStr={
        "num_vars":0,
        "names":[],
        "bounds":[]
    }

    paramsRangeFile=None
    analysisDir=None
    sampleFile=None
    jsonProblem=None
    paramValues=None
    paramRowCount=0
    nextRecord=0

    def generateSamples(self, analysisDir, samples=340):
        self.analysisDir=analysisDir
        self.jsonProblem=self.analysisDir+"/problem.json"
        self.paramsRangeFile=self.analysisDir+"/screening_inputparams.csv"
        self.paramsFile=self.analysisDir+"/params_"+'{0:05}'.format(0)+".csv"

        # if os.path.exists(self.paramsFile):
        #     self.paramValues=pd.read_csv(self.paramsFile)
        #     return self.paramValues
    
        jsonfile = open(self.jsonProblem, 'w')
        paramsRangeDF =  pd.read_csv(self.paramsRangeFile)
        print(paramsRangeDF)
        row_count = len(paramsRangeDF)
        self.problemStr["num_vars"]=row_count
        self.problemStr["names"]=paramsRangeDF['Parameter'].tolist()
        self.problemStr["bounds"]=[None] * row_count

        for i in range(row_count):
            self.problemStr["bounds"][i]=[paramsRangeDF.iloc[i][1].tolist(),paramsRangeDF.iloc[i][2].tolist()]
        jsonfile.write(json.dumps(self.problemStr,indent=4)) 
        self.paramValues = saltelli.sample(self.problemStr, samples)
        self.paramRowCount=self.paramValues.shape[0]
        csvHeader=str(self.problemStr['names']).strip('[] ').replace("'", "").replace(" ","")
        print(csvHeader)
        print(self.paramsFile)
        np.savetxt(self.paramsFile,self.paramValues,delimiter=',',header=csvHeader,comments='') 
        self.paramValues=pd.read_csv(self.paramsFile)
        self.paramValues.drop_duplicates()
        self.paramValues.to_csv(self.paramsFile,index=False)
        return self.paramValues

    def getNSamples(self,count):
        X=self.paramValues.iloc[self.nextRecord:count]
        self.nextRecord=self.nextRecord+count
        return X

if __name__ == '__main__':
    sobolSeq=DyMMMSamplesSobol()

    analysisDir=settings.simSettings["analysisDir"]

    if(len(sys.argv)>1):
        analysisDir=sys.argv[1]

    sobolSeq.generateSamples(analysisDir)
    #print(sobolSeq.getNSamples(30))

