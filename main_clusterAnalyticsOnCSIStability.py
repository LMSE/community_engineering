
import pandas as pd
import numpy as np
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import collections
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import sklearn
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import DyMMMSettings as settings
from sklearn.base import clone 
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid, train_test_split
import numpy as np

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#np.set_printoptions(threshold=sys.maxsize)

stateNames=['biomass1','biomass2','EX_glc__D_e','EX_his__L_e','A1','A2','R1','R2','G1','G2']


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



#============================================FEATURE IMPORTANCES===========================================================================================================

from sklearn.base import clone 

def dropcol_importances(rf, X_train, y_train):
    """
    Calculates the drop column feature importance.

    Parameters:
        rf (RandomForestClassifier): The classifier to use for calculating importances.
        X_train (DataFrame): Training feature set.
        y_train (Series): Training target set.

    Returns:
        DataFrame: A DataFrame containing feature importances.
    """
    rf_ = clone(rf)
    rf_.set_params(warm_start=True, oob_score=True)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)#.ravel())
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

def featureImportance(X, y, titleStr):
    """
    Calculates and plots feature importances using Extra Trees Classifier.

    Parameters:
        X (DataFrame): The feature set.
        y (Series): The target variable.
        titleStr (str): Title string for the plot.
    """
    feature_names=X.columns.to_list()
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=1200,
                                random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    colNames=[]

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        colNames.append(X.columns[indices[f]])

    print(colNames)

    df = pd.DataFrame(data=colNames, columns=['param'])
    df['importances']=importances[indices]
    df['std']=std[indices]
    print(df)
    plt.figure()
    plt.title("Feature importances "+titleStr)
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), colNames)
    plt.xlim([-1, X.shape[1]])
    #plt.show()

def featureImportance2(resultsPath, X, y, titleStr):
    """
    Calculates and plots feature importances using Random Forest and permutation importance.

    Parameters:
        resultsPath (str): Path to save the output plots.
        X (DataFrame): The feature set.
        y (Series): The target variable.
        titleStr (str): Title string for the plot.
    """
    feature_names=X.columns.to_list()

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=120,warm_start=True, max_features=None,
                               oob_score=True,
                                random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    colNames=[]

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        colNames.append(X.columns[indices[f]])

    #print(colNames)
    
    df = pd.DataFrame(data=colNames, columns=['param'])
    df['importances']=importances[indices]
    df['std']=std[indices]
    print(df)
    fig=plt.figure()
    plt.title("Feature importances (default random forrest) "+titleStr)
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), colNames)
    plt.xlim([-1, X.shape[1]])
    #plt.show()
    fig.savefig(resultsPath+'/'+titleStr+"_defaultRandomForrest.svg", format='svg')  

    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    print(forest_importances)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model "+titleStr)
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    #plt.show()
    fig.savefig(resultsPath+'/'+titleStr+"_permutation.svg", format='svg')  
    
    df=dropcol_importances(forest, X_train, y_train)
    fig=plt.figure()
    plt.title("Feature importances (drop column importances method) "+titleStr)
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), colNames)
    plt.xlim([-1, X.shape[1]])
    plt.show() 
    fig.savefig(resultsPath+'/'+titleStr+"_dropcol.svg", format='svg')  
  



#============================================CLUSTER ANALYTICS===========================================================================================================

def getDataDF(resultsPath, resultDF, communityName):
    eigenFile=resultsPath+communityName+'_stability.csv'
    if not os.path.exists(eigenFile):
        eigenFile=resultsPath+communityName+'_eigen.npy'

    if eigenFile.endswith('.npy'):
        eigenData=np.loadtxt(eigenFile).view(complex)
        eigenData=pd.DataFrame(data=eigenData, columns=stateNames)

    else:
        df=pd.read_csv(eigenFile)
        eigenData=pd.DataFrame(columns=stateNames)
        for columnName in eigenData.columns.to_list():
            if columnName not in ['vol']:
                eigenData[columnName] = df[columnName].str.replace('i','j').apply(lambda x: np.complex(x))
            else:
                eigenData[columnName] = df[columnName]

    #resultDF=resultDF[resultDF["CSI"] > 0.8]

    df = eigenData[eigenData.index.isin(resultDF.index)]

    df['CSI'] = resultDF["CSI"]
    df['community'] = communityName

    df['biomass1_real'] = df['biomass1'].apply(lambda r: r.real)
    df['biomass1_cplx'] = df['biomass1'].apply(lambda r: r.imag)
    df['biomass2_real'] = df['biomass2'].apply(lambda r: r.real)
    df['biomass2_cplx'] = df['biomass2'].apply(lambda r: r.imag)

    df['networkType'] = 0

    df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0), 'networkType']=1
    #df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] < 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='b'
    #df.loc[(df["biomass1_real"] < 0) & (df["biomass2_real"] > 0), 'networkType']='c'
    #df.loc[(df["biomass1_real"] > 0) & (df["biomass2_real"] < 0), 'networkType']='c'
    #df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0), 'networkType']='g'
    #df.loc[(df["biomass1_real"] >= 0) & (df["biomass2_real"] >= 0) & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3), 'networkType']='d'
    #df.loc[((df['networkType'] == 'a') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='e'
    #df.loc[((df['networkType'] == 'c') & (abs(df["biomass1_cplx"]) > 0) & (abs(df["biomass1_cplx"]) - abs(df["biomass2_cplx"]) < 1e-3)),('networkType')]='f'
    print(df)
    return df


def generateClusterSampleFiles(resultsPath, df_params, clusterListOfSampleList):
    """
    Generates files with samples for each cluster.

    Parameters:
        resultsPath (str): Path to save the cluster files.
        df_params (DataFrame): DataFrame containing parameters.
        clusterListOfSampleList (list): A list of lists, where each sublist represents a cluster.
    """
    for idx1, clusterList in enumerate(clusterListOfSampleList):
        df=pd.DataFrame(columns=df_params.columns.to_list())
        for idx2, rowId in enumerate(clusterList):
            df=df.append(df_params.loc[rowId])
        #print(df)
        clustersPath = resultsPath + "/clusters/"
        if not os.path.exists(clustersPath):
            os.makedirs(clustersPath)
        df.to_csv(clustersPath+str(idx1)+".csv", index=False)        

def generateClusterAnalytics(model):
    """
    Generates analytics for the clusters formed by DBSCAN.

    Parameters:
        model (DBSCAN): The DBSCAN clustering model.

    Returns:
        tuple: A tuple containing the list of noisy samples and a list of lists representing the clusters.
    """
    n_clusters_=model.labels_.max()
    print("Number of clusters ",n_clusters_)
    clusterListOfSampleList=[[] for i in range(n_clusters_)]
    noisySampleList=[]
    for idx, value in np.ndenumerate(model.labels_):
        if value==-1:
            noisySampleList.append(idx[0])
        else:
            clusterListOfSampleList[value-1].append(idx[0])
    return noisySampleList,clusterListOfSampleList

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

def getMinMaxRow(idx, df):
    if not os.path.exists('data/temp'):
        os.makedirs('data/temp')    
    filename="data/temp/"+str(idx)+".csv"
    df.to_csv(filename,index=False)
    maxValues=df.max()
    minValues=df.min()
    row={}
    row['id']=idx
    vol=1
    for col in df.columns.to_list():
        if col == 'CSI':
            continue
        row[col+'_L']=minValues[col]
        row[col+'_H']=maxValues[col]
        vol*=maxValues[col]-minValues[col]
    row['vol']=vol
    return row

def getClusterParamRange(noisySampleList, clusterListOfSampleList, X, colNames, fileName):
    csvdf=pd.DataFrame()
    listOfDf=[]
    df=pd.DataFrame(columns=colNames)
    for idx in noisySampleList:
        df=df.append(X.loc[idx])
    listOfDf.append(df)

    for samplesList in clusterListOfSampleList:
        df=pd.DataFrame(columns=colNames)
        for idx in samplesList:
            df=df.append(X.loc[idx])
        listOfDf.append(df)

    for idx, df in enumerate(listOfDf):
        print("-------------",idx,"----------------------")
        #print(df.apply(minMax))
        rowDict=getMinMaxRow(idx, df)
        num=df.loc[df['CSI'] == 1].shape[0]
        purity=float(num/df.shape[0])
        rowDict['purity']=purity
        num=df.loc[df['stability'] == 1].shape[0]
        purity=float(num/df.shape[0])
        rowDict['stability']=purity        
        rowDict['count']=df.shape[0]
        #print(rowDict)
        csvdf=csvdf.append(rowDict, ignore_index=True)
    
    print("output in file: "+fileName)
    csvdf.to_csv(fileName, index=False)


#--------------------------------------------DBSCAN------------------------------------------------------


if __name__ == '__main__':
    
    communitiesDir=settings.simSettings["communitiesDir"]
    communityName=settings.simSettings["communityName"]
    analysisDir=settings.simSettings["analysisDirName"]

    screeningParamFiles = communitiesDir+"/"+communityName+"/screening_inputparams.csv"


    minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(screeningParamFiles)

    resultsPath = communitiesDir+"/"+communityName+"/results/"
    df_params=pd.read_csv(resultsPath+communityName+"_params.csv")
    FEATURE_NAMES=df_params.columns.tolist()
    #print(df_params)
    df_result=pd.read_csv(resultsPath+communityName+"_results.csv")

    #print(df_result)

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = df_params.copy()
    X=applyScaler(X, scaler)
    X=pd.DataFrame(data=X,columns=FEATURE_NAMES)

    target=df_result['CSI']
    target.loc[target >= 0.8] = 1
    target.loc[target < 0.8] = 0

    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_Y = encoder.transform(target)
    y=encoded_Y

    print(X)
    print(y)
    #uncomment following lines to get heighbour score - https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
    #neighbors = NearestNeighbors(n_neighbors=20)
    #neighbors_fit = neighbors.fit(X)
    #distances, indices = neighbors_fit.kneighbors(X)
    #distances = np.sort(distances, axis=0)
    #distances = distances[:,1]
    #plt.plot(distances)
    #plt.show()
    #quit()

    """

    # Define a range of hyperparameters to search over
    param_grid = {'eps': np.linspace(0.5, 2, 10), 'min_samples': np.arange(10, 15)}

    # Create a grid of hyperparameters to evaluate
    param_grid = ParameterGrid(param_grid)

    # Evaluate DBSCAN with different hyperparameters and choose the best ones
    best_score = -1
    for params in param_grid:
        print(params)
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        dbscan.fit(X)
        labels = dbscan.labels_
        score = -1
        try:
            score = silhouette_score(X, labels)
        except Exception as e:
            print(e)
            continue
        print(score)
        if score > best_score:
            best_score = score
            best_params = params
            print("Best score and params ", str(best_score), str(best_params))

    # Fit DBSCAN with the best hyperparameters on the entire dataset
    print("Selected best score and params ", str(best_score), str(best_params))
    model = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    """
    model = DBSCAN(eps=1.0, min_samples=14) #for prediso
    #model = DBSCAN(eps=0.5, min_samples=14) #for prednew
    model.fit(X)
    

    noisySampleList,clusterListOfSampleList=generateClusterAnalytics(model)

    print(clusterListOfSampleList)

    generateClusterSampleFiles(resultsPath, df_params, clusterListOfSampleList)

    df=getDataDF(resultsPath, df_result, communityName)

    X['CSI']=y
    df_params['CSI']=y
    df_params['stability']=df['networkType']

    FEATURE_NAMES.append('CSI')
    FEATURE_NAMES.append('stability')

    #getClusterParamRange(noisySampleList, clusterListOfSampleList, X, FEATURE_NAMES, resultsPath+"/ClusterAnalyticsNormalized.csv")
    #getClusterParamRange(noisySampleList, clusterListOfSampleList, df_params, FEATURE_NAMES, resultsPath+"/ClusterAnalytics.csv")


    #---- uncomment for Feature---------------------
    
    X.drop('CSI', inplace=True, axis=1)
    y1=pd.DataFrame(data=y, columns=['CSI'])
    y1.loc[y1['CSI'] >= 0.8, 'CSI'] = 1
    y1.loc[y1['CSI'] < 0.8, 'CSI'] = 0
    #featureImportance2(resultsPath, X, y1, 'Diversity')
    df=getDataDF(resultsPath, df_result, communityName)
    featureImportance2(resultsPath, X, df['networkType'], 'Stability - high CSI')

