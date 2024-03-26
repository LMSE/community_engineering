import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import plotly.graph_objects as go
import decimal
import math
import DyMMMSettings as settings


"""
visualizes the distribution and overlap of parameter values across different clusters identified in the simulation results. Processes 
the cluster data, normalizes the parameter values, and uses Plotly to create a layered plot that provides insights into the parameter 
ranges for each cluster, facilitating an understanding of the diversity within the clusters.
"""

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)

communitiesDir=settings.simSettings["communitiesDir"]
communityName="communitypred_cstr"
analysisDir=settings.simSettings["analysisDirName"]

inputFile = communitiesDir+"/"+communityName+"/results/ClusterAnalytics.csv"
paramsFile = communitiesDir+"/"+communityName+"/screening_inputparams.csv"

df=pd.read_csv(inputFile)

fileList=[]

print(df)

#df=df[df["purity"] == 1.0]
#df=df[df["count"] > 2*17]


print(df)

fileList=[]

clustersPath=communitiesDir+"/"+communityName+"/results/clusters/"
for index, row in df.iterrows():
    if index == 0:
        continue
    filePath=clustersPath+str(row['id'] - 1)[0:-2]+'.csv'
    print(filePath)
    fileList.append(filePath)

print(fileList)

def generateRangesScalar(paramsRangeFile):
    paramsRangeFileDf=pd.read_csv(paramsRangeFile)
    minValueRange=paramsRangeFileDf['MinValue'].tolist()
    maxValueRange=paramsRangeFileDf['MaxValue'].tolist()
    scaler=[MinMaxScaler() for i in range(len(minValueRange))]
    [scaler[i].fit([[minValueRange[i]], [maxValueRange[i]]]) for i in range(len(minValueRange))]
    return minValueRange, maxValueRange, scaler, paramsRangeFileDf

def applyScaler(X, scaler):
    X_n=np.copy(X)
    for i in range(X.shape[1]):
        #print(X.iloc[:,i].to_numpy().reshape(-1,1))
        X_n[:,i]=scaler[i].transform(X.iloc[:,i].to_numpy().reshape(-1,1))[:,0]
    return(X_n)

minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(paramsFile)

X=None

init=True
for inputFile in fileList:
    print(inputFile)
    X1=pd.read_csv(inputFile)
    #X1.drop(['CSI'], inplace=True, axis=1)
    FEATURE_NAMES = X1.columns.tolist()
    X1=pd.DataFrame(data=applyScaler(X1, scaler), columns=FEATURE_NAMES)
    if init==True:
        X=X1
        init=False
    else:
        print("Appending===================")
        X=X.append(X1, ignore_index=True)
    print(X)

print(X)
x_axis=list(float_range(0,1.01,'0.01'))
occ_data=pd.DataFrame(0, index=x_axis, columns=FEATURE_NAMES)

X=X.round(2)

for param in FEATURE_NAMES:
    counts=X[param].value_counts()
    #print(counts.max())    
    counts=counts/counts.max()
    for index, item in counts.iteritems():
        occ_data.at[index,param]=item

fig = go.Figure()
for index, paramName in enumerate(FEATURE_NAMES):

    fig.add_trace(go.Scatter(
                            x=x_axis, y=np.full(1, len(FEATURE_NAMES)-index),
                            mode='lines',
                            line_color='white'))
    
    fig.add_trace(go.Scatter(
                            x=x_axis,
                            y= occ_data[paramName] + (len(FEATURE_NAMES)-index),
                            fill='tonexty',
                            name=f'{paramName}'))

    # plotly.graph_objects' way of adding text to a figure
    fig.add_annotation(
                        x=0,
                        y=len(FEATURE_NAMES)-index,
                        text=f'{paramName}',
                        showarrow=False,
                        xshift=20,
                        yshift=10)

# here you can modify the figure and the legend titles
fig.update_layout(
                title='High Diversity Cluster Parameter Ranges',
                showlegend=False,
                xaxis=dict(title='Normalized Parameter Range'),
                yaxis=dict(showticklabels=False) # that way you hide the y axis ticks labels
                )

fig.show()

