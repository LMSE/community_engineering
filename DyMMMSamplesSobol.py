import os
import sys

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import DyMMMSettings as settings
from SALib.sample import saltelli

"""The code generates the space filing samples in the specified directory."""
"""copy or create the screening_inputparams.csv"""

class DyMMMSamplesSobol:
    """
    Class to generate Sobol sequence samples for parameters defined in a CSV file.
    
    Attributes:
        problemStr (dict): Dictionary containing the problem definition for Sobol sampling.
        paramsRangeFile (str): Path to the CSV file containing the parameter ranges.
        analysisDir (str): Directory where the analysis is being conducted.
        sampleFile (str): Path to the file where generated samples will be saved.
        jsonProblem (str): Path to the JSON file where the problem definition is stored.
        paramValues (DataFrame): DataFrame containing the generated parameter values.
        paramRowCount (int): Number of rows in the generated parameter values DataFrame.
        nextRecord (int): Index of the next record to retrieve in a batch of samples.
    """
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

    def generateSamples(self, analysisDir, samples=64):
        """
        Generates Sobol sequence samples based on the parameter ranges specified in a CSV file.
        
        Parameters:
            analysisDir (str): Directory where the analysis is being conducted and files are stored.
            samples (int, optional): Number of samples to generate. Default is 64.
        
        Returns:
            DataFrame: A DataFrame containing the generated Sobol sequence samples.
        """
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
        """
        Retrieves a specified number of samples from the generated samples.
        
        Parameters:
            count (int): Number of samples to retrieve.
        
        Returns:
            DataFrame: A subset of the generated samples DataFrame.
        """
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

