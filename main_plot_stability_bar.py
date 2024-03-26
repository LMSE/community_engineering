

"""
functions to process and visualize the data related to the stability and diversity of biological communities based on their eigenvalues 
and other parameters. It includes functionalities to extract complex values, generate data frames for analysis, and create BAR plots to 
visualize the data.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import DyMMMSettings as settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

stateNames=['vol','biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']


def plotEigen(community, diversity, df):
    for param in ['biomass1', 'biomass2']:
        index=['biomass1', 'biomass2'].index(param)
        X = df[param].apply(lambda r: r.real)
        Y = df[param].apply(lambda r: r.imag)
        plt.scatter(X,Y, marker=".")
        plt.title("{} Diversity in {} for {} ".format(diversity, community, param))
    plt.show()
    plt.figure().savefig(community+" eigen.svg", format='svg')  


def getCplxValues(eigenData, fileName, paramsDF):
    df=eigenData.copy()
    #df1=pd.DataFrame(columns=['row','Y1', 'Y2', 'DIFF'])
    df1=pd.DataFrame(columns=stateNames)
    df2=pd.DataFrame(columns=paramsDF.columns.to_list())
    matched=0
    mismatched=0
    mismatched_cplx=0
    matched_cplx=0
    for index, row in df.iterrows():
        AddParamRow=False
        X1 = row['biomass1'].real
        X2 = row['biomass2'].real
        Y1 = row['biomass1'].imag
        Y2 = row['biomass2'].imag

        if(abs(Y1) > 0 or abs(Y2) > 0):
            if(X1-X2< 1e-3):
                matched+=1
            else:
                mismatched+=1
            if(Y1+Y2< 1e-3):
                matched_cplx=matched_cplx+1
            else:
                mismatched_cplx=mismatched_cplx+1
                AddParamRow=True
        if AddParamRow==True:
            #df1=df1.append({'row':index,'Y1':Y1, 'Y2':Y2 , 'DIFF':Y1+Y2}, ignore_index=True)
            df1=df1.append(row, ignore_index=True)
            df2=df2.append(paramsDF.iloc[index], ignore_index=True)
    df1.to_csv(fileName, index=False)
    df2.to_csv(fileName[:-4]+"_params.csv", index=False)
    print(fileName , "rows matched mismatched matched_cplx mismatched_cplx ", df.shape[0], matched, mismatched, matched_cplx, mismatched_cplx)


def getCplxValuesExt(eigenData, fileName, paramsDF):
    df=eigenData.copy()
    with open(fileName, 'w') as filehandle:    
        for index, row in df.iterrows():
            info=[]
            for idx, x in np.ndenumerate(row):
                if (abs(x.imag) > 0):
                    info.append({stateNames[idx[0]]: x.imag})
            if len(info) > 0:
                #print(index, info)
                filehandle.write('%s\n' % info)


def getDataDF(resultsPath, communityName):
    paramFile=resultsPath+communityName+'_params.csv'
    resultFile=resultsPath+communityName+'_results.csv'  
    eigenFile=resultsPath+communityName+'_eigen.csv'
    if not os.path.exists(eigenFile):
        eigenFile=resultsPath+communityName+'_eigen.npy'

    paramDF=pd.read_csv(paramFile)

    resultDF=pd.read_csv(resultFile)

    if eigenFile.endswith('.npy'):
        eigenData=np.loadtxt(eigenFile).view(complex)
        eigenData=pd.DataFrame(data=eigenData, columns=stateNames)

    else:
        df=pd.read_csv(eigenFile)
        eigenData=pd.DataFrame(columns=stateNames)
        for columnName in eigenData.columns.to_list():
            eigenData[columnName] = df[columnName].str.replace('i','j').apply(lambda x: np.complex(x))

    #resultDF=resultDF[resultDF["CSI"] > 0.8]

    df = eigenData[eigenData.index.isin(resultDF.index)]

    df['CSI'] = resultDF["CSI"]
    df['community'] = communityName

    df['biomass1_real'] = df['biomass1'].apply(lambda r: r.real)
    df['biomass1_cplx'] = df['biomass1'].apply(lambda r: r.imag)
    df['biomass2_real'] = df['biomass2'].apply(lambda r: r.real)
    df['biomass2_cplx'] = df['biomass2'].apply(lambda r: r.imag)

    df['networkType'] = 'x'

    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0), 'networkType']='a'
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='b'
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] > 0), 'networkType']='c'
    df.loc[(df["biomass1_real"] > 0) & (df["biomass2_real"] < 0), 'networkType']='c'
    df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0), 'networkType']='g'
    df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='d'
    df.loc[((df['networkType'] == 'a') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='e'
    df.loc[((df['networkType'] == 'c') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='f'
    print(df.describe())
    return df, paramDF, resultDF


def plotData(CSIType, resultsPath, communityName):

    df1, paramDF, resultDF=getDataDF(resultsPath, communityName)
    df1=df1[0:53000]
    if CSIType[0]=='H':
        df1=df1.loc[df1['CSI'] > 0.8]
    elif CSIType[0]=='L':
        df1=df1.loc[df1['CSI'] < 0.2]

    # a1=df1['networkType'].loc[df1['networkType']=='a'].count()
    # print(a1)

    networkType=['a', 'b', 'c', 'd', 'e', 'f']

    dataDict={}
    for idx in range(len(networkType)):
        networkId=networkType[idx]
        dataDict[networkId]= [
            df1['networkType'].loc[df1['networkType']==networkId].count(),
        ]

    #print(dataDict)

    plotdata = pd.DataFrame(
        dataDict,index=["pred"]
    )

    #ax = plt.axes()
    #ax.set_yticklabels([])
    ax1 = plotdata.plot(kind="bar")
    plt.yticks([])
    plt.title(CSIType+" Stability")
    plt.ylabel("samples")
    plt.show()
    plt.figure().savefig(CSIType+" Stability.svg", format='svg')  


def plotClusterData(CSIType, resultsPath, communityName):

    df1=getDataDF(resultsPath, communityName)
    if CSIType[0]=='H':
        df1=df1.loc[df1['CSI'] > 0.8]
    elif CSIType[0]=='L':
        df1=df1.loc[df1['CSI'] < 0.2]

    networkType=['a', 'b', 'c', 'd', 'e', 'f']

    dataDict={}
    for idx in range(len(networkType)):
        networkId=networkType[idx]
        dataDict[networkId]= [
            df1['networkType'].loc[df1['networkType']==networkId].count()      
        ]

    #print(dataDict)

    plotdata = pd.DataFrame(
        dataDict,index=["pred"]
    )

    #ax = plt.axes()
    #ax.set_yticklabels([])
    ax1 = plotdata.plot(kind="bar")
    plt.yticks([])
    plt.title(CSIType+" Stability")
    plt.ylabel("samples")
    plt.show()
    plt.figure().savefig(CSIType+" Stability.svg", format='svg')  


def plotDecisionTree(CSIType, resultsPath, communityName):

    df1, paramDF, resultDF=getDataDF(resultsPath, communityName)
    if CSIType[0]=='H':
        df1=df1.loc[df1['CSI'] > 0.8]
    elif CSIType[0]=='L':
        df1=df1.loc[df1['CSI'] < 0.2]
    FEATURE_NAMES=paramDF.columns.to_list()
    X_train=paramDF
    y_train=df1['networkType']

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train=encoded_Y
    #print(y_train)

    clf = DecisionTreeClassifier(#max_depth = 3,
                                random_state = 0)
    clf.fit(X_train, y_train)

    #get number
    leafCount=clf.get_n_leaves()
    print(leafCount)

    print("\nRULES output written to decision_tree.log")

    text_representation = tree.export_text(clf, feature_names=FEATURE_NAMES,show_weights=True)
    with open(resultsPath+"/decision_tree.log", "w") as fout:
        fout.write(text_representation)

    #fig = plt.figure(figsize=(25,20))
    plt.figure(figsize=(24,24))  # set plot size (denoted in inches)
    tree.plot_tree(clf, feature_names=FEATURE_NAMES, class_names='networkType', filled=True, fontsize=10)
    plt.show()


#plotDecisionTree()
communitiesDir=settings.simSettings["communitiesDir"]
communityName="communitypred_cstr"
analysisDir=settings.simSettings["analysisDirName"]
resultsPath=communitiesDir+"/"+communityName+"/results/"

plotData(" ",resultsPath,"communitypred_cstr")
plotData('High Diversity',resultsPath ,"communitypred_cstr")
plotData('Low Diversity',resultsPath,"communitypred_cstr")

plotClusterData("High CSI Cluster ",resultsPath , "communitypred_cstr")


