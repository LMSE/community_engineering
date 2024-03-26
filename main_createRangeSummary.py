import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ydata_profiling import ProfileReport
import sweetviz as sv

import DyMMMSettings as settings


"""
Aggregates parameter ranges across different simulation results, compiles them into a single dataframe, and generates 
a report using Sweetviz. It iteratively reads min-max range values from CSV files in a specified directory, constructs a dataframe 
from these values, and then produces an HTML report summarizing the data distribution and statistics.
"""


communitiesDir=settings.simSettings["communitiesDir"]
communityName=settings.simSettings["communityName"]
communityDir=communitiesDir+'/'+communityName+'/'
resultsDir=communityDir+'/results/'

outputFolder=resultsDir+'cluster_ranges'

def generateRangesScalar(paramsRangeFile):
    paramsRangeFileDf=pd.read_csv(paramsRangeFile)
    minValueRange=paramsRangeFileDf['MinValue'].tolist()
    maxValueRange=paramsRangeFileDf['MaxValue'].tolist()
    scaler=[MinMaxScaler() for i in range(len(minValueRange))]
    [scaler[i].fit([[minValueRange[i]], [maxValueRange[i]]]) for i in range(len(minValueRange))]
    return minValueRange, maxValueRange, scaler, paramsRangeFileDf

paramsRangeFileDf=pd.read_csv(settings.simSettings["communityDir"]+"/screening_inputparams.csv")

colList=[]

for index, row in paramsRangeFileDf.iterrows():
    paramName=row['Parameter']
    colList.append(paramName+"_Min")
    colList.append(paramName+"_Max")

rangeType='HighCSIRange'

print(colList)
data_df = pd.DataFrame(columns = colList)
searchStr=outputFolder+'/{}*'.format(rangeType)
files = glob.glob(searchStr)
for f in files:
    print(f)
    minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(f)
    rowItemList=[]
    for i in range(len(minValueRange)):
        rowItemList.append(minValueRange[i])
        rowItemList.append(maxValueRange[i])
    data_df.loc[len(data_df)]=rowItemList

print(data_df)
fileName=resultsDir+"{}Summary.csv".format(rangeType)
data_df.to_csv(fileName,index=False)
#profile = ProfileReport(data_df, title='Pandas Profiling Report', explorative=True)
#profile.to_file(fileName+"_profile.html")
analysis = sv.analyze(data_df)
analysis.show_html(fileName+"_sv.html")

