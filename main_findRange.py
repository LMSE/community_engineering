
import math
import numpy as np
import pandas as pd
import DyMMMSettings as settings
from scipy.spatial import distance

import pandas as pd
from pymoo.factory import get_performance_indicator
from sklearn.preprocessing import MinMaxScaler

"""
Provides functionalities to process and analyze community data. It includes scaling parameter data, calculating distance matrices between 
high and low CSI (Community Stability Index) points, determining parameter ranges, and analyzing these ranges to compute the volume of the
parameter space. The main goal is to understand the parameter space's impact on community stability based on the CSI.
"""

def processCommunity(communitiesDir, communityName, param_df, resultFile):
    """
    Processes community data by scaling parameters, computing distance matrices,
    and determining parameter ranges based on community stability indicators.

    Parameters:
        communitiesDir (str): Directory containing community data.
        communityName (str): Name of the community being processed.
        param_df (DataFrame): DataFrame containing parameters to be processed.
        resultFile (str): Path to the file containing community result data.

    Returns:
        list: List of parameter names processed.
    """
    x = param_df.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    param_df = pd.DataFrame(data=x_scaled, columns=param_df.columns)
    print(param_df)

    result_df=pd.read_csv(resultFile)

    result_df_A=result_df.loc[result_df['CSI'] >= 0.9]
    result_df_B=result_df.loc[result_df['CSI'] < 0.9]
    result_df_B=result_df_B.loc[result_df_B['CSI'] > 0.8]


    param_df_A = param_df[param_df.index.isin(result_df_A.index)]

    param_df_B = param_df[param_df.index.isin(result_df_B.index)]

    print("------------Ranges in param_df_A/B start-------------")
    print(param_df_A.shape)
    print(param_df_B.shape)
    print("------------Ranges in param_df_A/B end-------------")

    param_rows, param_cols=param_df_A.shape
    print(param_df_B.shape)
    param_B_rows, param_B_cols=param_df_B.shape
    
    paramList=param_df_B.columns.tolist()
    paramCount = len(paramList)
    print(paramCount)

    result=distance.cdist(param_df_A, param_df_B, 'cityblock')
    print('--------distance shape--------------')
    print(result.shape)
    rows, cols = result.shape

    paramRange=np.zeros(shape=(2, rows, paramCount))
    paramRangeScaled=np.zeros(shape=(2, rows, paramCount))

    #for each row of data get the min and max params
    for row in range(rows):
        #for each row in param_df_A
        point=param_df_A.iloc[row,:]
        sortedIndex=np.argsort(result[row,:])
        for col in range(cols):
            #get closest row from param_df_B
            currentRow = param_df_B.iloc[sortedIndex[col]]
            #for each param of closest row
            for paramIndex in range(paramCount):
                value=currentRow.iloc[paramIndex]
                #if lower param value is zero or greater - replace the value
                if paramRange[0, row, paramIndex] == 0 and point.iloc[paramIndex] > value:
                        paramRange[0, row, paramIndex]=value
                #if higher param value is zero or lesser - replace the value
                elif paramRange[1, row, paramIndex] == 0 and point.iloc[paramIndex] < value:
                        paramRange[1, row, paramIndex]=value

    #all rows min params for closest low CSI points
    paramRangeScaled[0]=min_max_scaler.inverse_transform(paramRange[0])
    #all rows max params for closest low CSI points
    paramRangeScaled[1]=min_max_scaler.inverse_transform(paramRange[1])

    np.save("{}/{}_rangesNormalized".format(communitiesDir, communityName), paramRange)
    np.save("{}/{}_ranges".format(communitiesDir, communityName), paramRangeScaled)

    # print("------------Ranges in numpy start-------------")
    # print(paramRange[0].shape)
    # print(paramRange[1].shape)
    # print("------------Ranges in numpy end-------------")


    return paramList
    

def analyzeRanges(communitiesDir, param_df):
    """
    Analyzes the parameter ranges to identify unique rows and compute the volume
    of the parameter space that corresponds to the maximum volume.

    Parameters:
        communitiesDir (str): Directory containing the community data.
        param_df (DataFrame): DataFrame containing the parameters for analysis.
    """
    paramRangeNormalized=np.load("{}/{}_rangesNormalized.npy".format(communitiesDir,communityName))
    paramRange=np.load("{}/{}_ranges.npy".format(communitiesDir, communityName))

    print(paramRange.shape)
    unique_rows = np.unique(paramRangeNormalized[0], axis=0)
    print(unique_rows.shape)
    unique_rows = np.unique(paramRangeNormalized[1], axis=0)
    print(unique_rows.shape)    
    vol=1
    maxVolIndex=-1
    maxVol=0
    #for each row 
    for row in range(paramRangeNormalized.shape[1]):
        vol=1
        for paramIndex in range(paramRangeNormalized.shape[2]):
            vol *= abs(paramRangeNormalized[1,row,paramIndex] - paramRangeNormalized[0,row,paramIndex])
        print(vol)
        if vol > maxVol:
            maxVol=vol
            maxVolIndex=row
    print(maxVolIndex)
    print(paramRange[0,maxVolIndex])
    print(paramRange[1,maxVolIndex])
    columnList=param_df.columns.tolist()
    for i in range(len(columnList)):
        print("{},{},{}".format(columnList[i], str(paramRange[0,maxVolIndex,i]), str(paramRange[1,maxVolIndex,i])))
    print("Normalized hypervolume "+ str(maxVol))



if __name__ == '__main__':

    analysisDir=settings.simSettings["analysisDir"]
    communityName=settings.simSettings["communityName"]
    paramsRangeFile=analysisDir+"/screening_inputparams.csv"
    communitiesDir=settings.simSettings["communitiesDir"]

    #minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(paramsRangeFile)
    #print(generateRangesScalar(paramsRangeFile))

    communityName="communitycoop_cstr"
    inputFile = communitiesDir+"/"+communityName
    paramFile = inputFile+"_param.csv"
    resultFile = inputFile+"_RESULT.csv"
    param_df=pd.read_csv(paramFile)
    processCommunity(communitiesDir, communityName, param_df, resultFile)
    analyzeRanges(communitiesDir, param_df)

