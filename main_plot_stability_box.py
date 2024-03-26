

"""
functions to process and visualize the data related to the stability and diversity of biological communities based on their eigenvalues 
and other parameters. It includes functionalities to extract complex values, generate data frames for analysis, and create BOX plots to 
visualize the data.
"""

import os
import sys
import glob
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from numpy import median
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import DyMMMSettings as settings
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

#stateNames=['vol','biomass1','biomass2','EX_glc__D_e','EX_his__L_e','EX_trp__L_e','A1','A2','R1','R2','G1','G2']
stateNames=['biomass1','biomass2']

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


def printDFStats(df, communityType, comment):
    print("\n\n\n\n\n\n\n\n\n--------------------statistics  "+communityType+"----------------------------------------------")
    df_a=df.loc[df['networkType']=='a']
    print(df.shape, df_a.shape)
    print(df_a.describe())
    printStabilityPercentage(df, communityType,comment)


def printStabilityPercentage(df, communityType, comment):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}={}====================================================".format(communityType, comment))
    a_count=df['networkType'].loc[df['networkType']=='a'].count()
    b_count=df['networkType'].loc[df['networkType']=='b'].count()
    c_count=df['networkType'].loc[df['networkType']=='c'].count()
    d_count=df['networkType'].loc[df['networkType']=='d'].count()
    e_count=df['networkType'].loc[df['networkType']=='e'].count()
    f_count=df['networkType'].loc[df['networkType']=='f'].count()
    g_count=df['networkType'].loc[df['networkType']=='g'].count()
    valueList=[df.shape[0], a_count, b_count, c_count, d_count, e_count, f_count, g_count]
    print(valueList)
    valueListpercent=[df.shape[0]/df.shape[0], a_count/df.shape[0], b_count/df.shape[0], c_count/df.shape[0], d_count/df.shape[0], e_count/df.shape[0], f_count/df.shape[0], g_count/df.shape[0]]
    print(valueListpercent)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


def plotDf(community, CSIType, df):
    
    allcols=['vol', 'biomass1', 'biomass2', 'EX_glc__D_e', 'EX_his__L_e', 'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2', 'CSI', 'community', 'biomass1_real', 'biomass1_cplx', 'biomass2_real', 'biomass2_cplx', 'networkType']
    dropcols=['vol', 'biomass1', 'biomass2', 'EX_glc__D_e', 'EX_his__L_e', 'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2', 'CSI']

    df.drop(dropcols, inplace=True, axis=1)
    print(df.columns.to_list())
    
    mdf = pd.melt(df, id_vars=['community'])

    print(mdf)

    boxplot = sns.boxplot(x="community", y="value", hue="Number", data=mdf)
    boxplot.axes.set_title(CSIType+ " Diversity Stability for "+community, fontsize=16)
    boxplot.set_xlabel("State", fontsize=14)
    boxplot.set_ylabel("Values", fontsize=14)
    plt.show()
    plt.savefig(CSIType+" Stability_box_{}_{}.svg".format(community, CSIType), dpi=300)  

def prepareDataFrame(paramDF, resultDF, eigenData, stabilityDF, communityType):

    stableRowsDF = stabilityDF.loc[(stabilityDF['errro_biomass1'] < 1e-3) & (stabilityDF['errro_biomass2'] < 1e-3 )  ]

    results_stableDF = resultDF[resultDF.index.isin(stableRowsDF.index)] # & (resultDF['biomass1'] > 0.1) & (resultDF['biomass2'] > 0.1 )]

    print(communityType, "total rows ", resultDF.shape, "stable rows", results_stableDF.shape)

    df = eigenData.loc[eigenData.index.isin(results_stableDF.index)]
    print(df)
    #df.drop(['ss_biomass1','ss_biomass2','errro_biomass1','errro_biomass2','delta','last_biomass1','max_biomass1','last_biomass2','max_biomass2','last_EX_glc__D_e'], inplace=True, axis=1)
    
    print(df)

    df['CSI'] = results_stableDF["CSI"].values
    df['community'] = communityType
    return df

def getDataDF(dataPath, communityType):
    paramFile=dataPath+communityType+'_params.csv'
    resultFile=dataPath+communityType+'_results.csv'  
    stabilityFile=dataPath+communityType+'_stability.csv'


    paramDF=pd.read_csv(paramFile)
    paramDF=paramDF[0:53000]

    resultDF=pd.read_csv(resultFile)
    resultDF=resultDF[0:53000]

    stabilityDF=pd.read_csv(stabilityFile)    
    stabilityDF=stabilityDF[0:53000]


    df=pd.read_csv(stabilityFile)
    df=df[0:53000]
    collist=['biomass1','biomass2','EX_glc__D_e','EX_his__L_e','A1','A2','R1','R2','G1','G2','ss_biomass1','ss_biomass2','errro_biomass1','errro_biomass2','delta','last_biomass1','max_biomass1','last_biomass2','max_biomass2','last_EX_glc__D_e','Vol','EX_ile__L_e']
    stateNames=['biomass1','biomass2']
    eigenData=pd.DataFrame(columns=stateNames)
    for columnName in df.columns.to_list():
        if columnName in stateNames:
            eigenData[columnName] = df[columnName].str.replace('i','j').apply(lambda x: np.complex(x))

    df=prepareDataFrame(paramDF, resultDF, eigenData, stabilityDF, communityType)

    df['biomass1_real'] = df['biomass1'].apply(lambda r: r.real)
    df['biomass1_cplx'] = df['biomass1'].apply(lambda r: r.imag)    
    df['biomass2_real'] = df['biomass2'].apply(lambda r: r.real)
    df['biomass2_cplx'] = df['biomass2'].apply(lambda r: r.imag)
    #df['biomass1_phasorMag'] = df['biomass1'].apply(lambda r: math.sqrt(r.real*r.real+r.imag*r.imag))
    #df['biomass2_phasorMag'] = df['biomass2'].apply(lambda r: math.sqrt(r.real*r.real+r.imag*r.imag))

    df['networkType'] = 'x'
    """
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0) & (abs(df["biomass1_cplx"]) == 0) &(abs(df["biomass2_cplx"]) == 0), 'networkType']='a'
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='b'
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] > 0), 'networkType']='c'
    df.loc[(df["biomass1_real"] > 0) & (df["biomass2_real"] < 0), 'networkType']='c'
    df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0), 'networkType']='g'
    df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='d'
    df.loc[((df['networkType'] == 'a') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='e'
    df.loc[((df['networkType'] == 'c') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='f'
    """
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0) & (abs(df["biomass1_cplx"]) == 0) &(abs(df["biomass2_cplx"]) == 0), 'networkType']='a'
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='b'
    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] > 0)& (abs(df["biomass1_cplx"]) == 0) &(abs(df["biomass2_cplx"]) == 0), 'networkType']='c'
    df.loc[(df["biomass1_real"] > 0) & (df["biomass2_real"] < 0)& (abs(df["biomass1_cplx"]) == 0) &(abs(df["biomass2_cplx"]) == 0), 'networkType']='c'
    df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0), 'networkType']='g'
    df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='d'
    df.loc[((df['networkType'] == 'a') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='e'
    df.loc[((df['networkType'] == 'c') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='f'
    
    print(df)

    return df, paramDF, resultDF


def plotData(CSIType, resultsPath, communityName):

    df1, paramDF, resultDF=getDataDF(resultsPath, communityName)
    print(df1)
    if CSIType[0]=='H':
        df1=df1.loc[df1['CSI'] >= 0.8]
    elif CSIType[0]=='L':
        df1=df1.loc[df1['CSI'] < 0.2]
    printDFStats(df1, 'pred', CSIType)

    allcols=['vol', 'biomass1', 'biomass2', 'EX_glc__D_e', 'EX_his__L_e', 'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2', 'CSI', 'community', 'biomass1_real', 'biomass1_cplx', 'biomass2_real', 'biomass2_cplx', 'networkType']
    dropcols=['biomass1', 'biomass2', 'EX_glc__D_e', 'EX_his__L_e', 'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2', 'CSI']  
    #dropcols=['biomass1', 'biomass2','CSI']  

    #chp df1.drop(dropcols, inplace=True, axis=1)
    print(df1.columns.to_list())

    stabilityTypeList=['a','b','c','d','e','f']



    for stabilityType in stabilityTypeList:

        df = df1.copy()

        #if stabilityType=='a':
        #    df.drop(['biomass1_cplx', 'biomass2_cplx'], inplace=True, axis=1)

        df.to_csv('outfile.csv',index=False)

        df = df.loc[df['networkType']==stabilityType]
        df.drop(['networkType'], inplace=True, axis=1)
        if df.shape[0] == 0:
            continue
        mdf = pd.melt(df, id_vars=['community'])

        print(mdf)

        boxplot = sns.boxplot(x='community', y="value", hue='variable', data=mdf, showfliers=False)
        boxplot.axes.set_title(CSIType+ " Diversity Stability Type=[{}]".format(stabilityType), fontsize=16)
        boxplot.set_xlabel("Community & States", fontsize=14)
        boxplot.set_ylabel("Values", fontsize=14)
        plt.show()
        boxplot.get_figure().savefig(resultsPath+'/'+CSIType+" Stability_box_{}_{}_{}.svg".format("pred", CSIType, stabilityType), dpi=300)  


    # plotDf('coop', CSIType, df1)
    # plotDf('comp', CSIType, df2)
    # plotDf('pred', CSIType, df3) 


def plotClusterData(CSIType):

    df1=getDataDF('data/cluster_results_pred_high/', 'pred')
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


def plotDecisionTree():

    df1, paramDF, resultDF=getDataDF('data/cluster_results_pred_high/', 'pred')
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
    with open("decision_tree.log", "w") as fout:
        fout.write(text_representation)

    #fig = plt.figure(figsize=(25,20))
    plt.figure(figsize=(24,24))  # set plot size (denoted in inches)
    tree.plot_tree(clf, feature_names=FEATURE_NAMES, class_names='networkType', filled=True, fontsize=10)
    plt.show()


#plotDecisionTree()
communitiesDir=settings.simSettings["communitiesDir"]
communityName=settings.simSettings["communityName"]
analysisDir=settings.simSettings["analysisDirName"]
resultsPath=communitiesDir+"/"+communityName+"/results/"

#print("\n\n\n=========================================================================FULL=============================================================================\n\n\n")
#plotData("ALL",resultsPath,communityName)
#print("\n\n\n=========================================================================HIGH DIVERSITY===================================================================\n\n\n")
#plotData('High Diversity',resultsPath ,communityName)
print("\n\n\n=========================================================================LOW DIVERSITY====================================================================\n\n\n")
plotData('Low Diversity',resultsPath,communityName)

#plotClusterData(" Cluster ")


