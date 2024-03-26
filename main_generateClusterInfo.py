

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import DyMMMSettings as settings

"""
Processes community data to identify parameter ranges that correspond to different stability indicators (e.g., high or low CSI).
It employs a Decision Tree Classifier to segment the parameter space into clusters, extracts the parameter ranges for each cluster, and 
computes the hypervolume for these ranges. The analysis results, including parameter ranges and associated data points for each cluster, are 
saved for further analysis.
"""


communitiesDir=settings.simSettings["communitiesDir"]
resultsDir=communitiesDir
communityName=settings.simSettings["communityName"]
communityDir=resultsDir+'/'+communityName+'/'

X_train_file=communityDir+'/results/'+communityName+'_params.csv'
y_train_file=communityDir+'/results/'+communityName+'_results.csv'

outputFolder=communityDir+'/results/cluster_ranges'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

files = glob.glob(outputFolder+'/*')
for f in files:
    os.remove(f)

def generateRangesScalar(paramsRangeFile):
    paramsRangeFileDf=pd.read_csv(paramsRangeFile)
    minValueRange=paramsRangeFileDf['MinValue'].tolist()
    maxValueRange=paramsRangeFileDf['MaxValue'].tolist()
    scaler=[MinMaxScaler() for i in range(len(minValueRange))]
    [scaler[i].fit([[minValueRange[i]], [maxValueRange[i]]]) for i in range(len(minValueRange))]
    return minValueRange, maxValueRange, scaler, paramsRangeFileDf

minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(settings.simSettings["communityDir"]+"/screening_inputparams.csv")

X_train=pd.read_csv(X_train_file)
FEATURE_NAMES = X_train.columns.tolist()

y_train=pd.read_csv(y_train_file)
y_train.drop(y_train.columns.difference(['CSI']), 1, inplace=True)
print("-----------------------------------------------------------")
print(X_train.shape)
print(y_train.shape)


y_train.loc[y_train['CSI'] < 0.9] = 0 
y_train.loc[y_train['CSI'] >= 0.9] = 1
# X_Train_embedded = TSNE(n_components=2).fit_transform(X_train)
# X=X_Train_embedded
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train=encoded_Y

print(y_train)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
import collections


def get_lineage(tree, feature_names, childID):
     """
     Retrieves the lineage or the path of decision rules from the root to the specified child node in a decision tree.

     Parameters:
          tree (DecisionTreeClassifier): The decision tree classifier.
          feature_names (list): List of feature names.
          childID (int): The ID of the child node.

     Returns:
          list: The lineage or list of decision rules from the root to the child node.
     """
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]
     parent=0
     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     
     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = []
               #print("Lineage-{}".format(str(child)))
          if child in left:
               #print("L-child {}".format(str(child)))
               parent = np.where(left == child)[0].item()
               split = '<='
          else:
               #print("R-child {}".format(str(child)))
               parent = np.where(right == child)[0].item()
               split = '>'

          if parent != 0:
               #lineage.append(("parentNodeID "+str(parent), split, threshold[parent], features[parent]))
               lineage.append("{} {} {}".format(features[parent],split,str(threshold[parent])))

          if parent == 0: # it is a leaf node
               lineage.reverse()
               return lineage
          else:
               ret=recurse(left, right, parent, lineage)
               return ret


     node=recurse(left, right, childID)
     return node
     #for child in idx:
     #     for node in recurse(left, right, child):
     #          print(node)



clf = DecisionTreeClassifier(#max_depth = 5,
                             random_state = 0)
clf.fit(X_train, y_train)

#https://mljar.com/blog/visualize-decision-tree/
# tree.plot_tree(clf)
# plt.show()

#get number
leafCount=clf.get_n_leaves()

#index of leaf for each sample
leafIndex=clf.apply(X_train)

print(leafCount)
print(leafIndex.shape)
print(leafIndex)
print(leafIndex.max())

leafMemberList= [] * leafIndex.max()
for i in range(leafIndex.max()):
        leafMemberList.append([])

#create a list of y-values for each cluster 
for index in range(leafIndex.shape[0]):
        #print("{} to index {}".format(str(y_train[index]),str(leafIndex[index])))
        leafMemberList[leafIndex[index]-1].append(y_train[index])

#print("=============================================================================")
#print(leafMemberList)


totalClusters=0
largetClusterHighCSIPoints=0
largestClusterID=0
highClusterList=[]
lowClusterList=[]

for i in range(leafIndex.max()):
     count=len(leafMemberList[i])
     if count == 0:
          continue
#     print("\n\n--------------CLUSTER-{}----------------------".format(i))
     lowCSICount=leafMemberList[i].count(0)
     highCSICount=leafMemberList[i].count(1)
#     print("Total Samples = {} lowCSISamplePoints={} highCSIPoints={}".format(str(count),str(lowCSICount),str(highCSICount)))
     if lowCSICount == 0 and highCSICount > 0:
          highClusterList.append((i+1, highCSICount)) #append tuple with index and count 
     if lowCSICount > 0 and highCSICount == 0:
          lowClusterList.append((i+1, lowCSICount))
     totalClusters+=1
     if largetClusterHighCSIPoints < highCSICount:
               largetClusterHighCSIPoints=highCSICount
               largestClusterID=i

print("---------------------------------------------------------------------")
print("------ClusterID are zero-indexed and LeafID are 1 indexed------------")
print("-------------------------RESULT--------------------------------------")
print("---------------------------------------------------------------------")
print("---------------------------------------------------------------------")
#print("Leaf {} and leaf with points {}".format(str(leafIndex.max()), str(totalClusters)))
print("LeafID {}  has cluster of {} highCSIPoints".format(str(largestClusterID+1),str(largetClusterHighCSIPoints)))

print("Note largest cluster point does not mean largest region but it is very likely. Every cluster of high CSI need to be checked")
print("\nRULES output written to decision_tree.log")

text_representation = tree.export_text(clf, feature_names=FEATURE_NAMES,show_weights=True)
with open("decision_tree.log", "w") as fout:
    fout.write(text_representation)


frequency = collections.Counter(leafIndex.tolist())
# printing the frequency
sortedLeafIndex = collections.OrderedDict(sorted(frequency.items()))
#print(sortedLeafIndex)


def getRange(leafid, print = False):
     """
     Extracts the parameter range for the given leaf node ID.

     Parameters:
          leafid (int): The ID of the leaf node.
          print (bool): Flag to print the range information.

     Returns:
          DataFrame: The parameter range for the specified leaf node.
     """
     node = get_lineage(clf, FEATURE_NAMES, leafid)
     df=pd.read_csv(settings.simSettings["communityDir"]+"/screening_inputparams.csv")
     for index in range(len(node)):
          if print:
               print(node[index])
          str_splits=node[index].split()
          rowIndex=df.index[df['Parameter'] == str_splits[0]].tolist()
          rowIndex=rowIndex[0]
          if str_splits[1] == '>':
               minValue=df.at[rowIndex, 'MinValue']
               if minValue < float(str_splits[2]):
                    df.at[rowIndex, 'MinValue'] = float(str_splits[2])
          else:
               maxValue=df.at[rowIndex, 'MaxValue']
               if maxValue > float(str_splits[2]):
                    df.at[rowIndex, 'MaxValue'] = float(str_splits[2])
     return df

def hyperVolume(df):
     """
     Calculates the hypervolume given the parameter ranges in a DataFrame.

     Parameters:
          df (DataFrame): DataFrame containing the parameter ranges.

     Returns:
          tuple: The scaled hypervolume and the original hypervolume.
     """
     hyperVolume=1
     hyperVolumeScaled=1
     for index, row in df.iterrows():
          maxValue=row['MaxValue']
          minValue=row['MinValue']
          arr = np.array([maxValue, minValue])
          data=scaler[index].transform(arr.reshape(-1,1))
          #data=scaler[index].transform([[maxValue,minValue]])
          hyperVolumeScaled*=data[0]-data[1]
          hyperVolume*=maxValue-minValue
     return hyperVolumeScaled, hyperVolume

#print("---------------------------------------Clusters with high CSI------tuple(leafid, number of points)----------------------------------------------------------")
highCSIList=sorted(highClusterList, key=lambda x: x[1], reverse=True)
#print(highClusterList)

#print("---------------------------------------Clusters with low CSI-------tuple(leafid, number of points)---------------------------------------------------------")
lowCSIList=sorted(lowClusterList, key=lambda x: x[1], reverse=True)
#print(lowClusterList)

print("\n\n----------------------------------------------------------------")
print("------------Range for Largest HIGH CSI clusters------------------")
high_df = pd.DataFrame(columns = ['c_index','normalized_hv','hv', 'samplePoints'])



y_train=pd.read_csv(y_train_file)
X_train['CSI']=y_train['CSI']

for i in range(len(highCSIList)):
     leafID=highCSIList[i][0] #ClusterID are zero-indexed and LeafID are 1 indexed
     samplePoints=highCSIList[i][1]
     range_df=getRange(leafID)
     volScaled,vol=hyperVolume(range_df)
     #print("index {} Hypervolume={} [{}]".format(str(i), str(volScaled),str(vol)))
     high_df = high_df.append({'c_index':leafID, 'normalized_hv':volScaled, 'hv':vol, 'samplePoints':samplePoints}, ignore_index=True)

     range_df.to_csv(outputFolder+'/HighCSIRange_{}.csv'.format(str(leafID)), index=False)

     if samplePoints > 0:
          data_df = pd.DataFrame(columns = X_train.columns.tolist())
          leafIndexList=leafIndex.tolist()
          for sampleIndex in  range(len(leafIndex)):
               if leafIndex[sampleIndex] == leafID:
                    data_df=data_df.append(X_train.iloc[[sampleIndex]])
                    #print(X_train.iloc[[sampleIndex]])
          data_df.to_csv(outputFolder+'/HighCSIData_{}.csv'.format(str(leafID)), index=False)


hvFileName=communityDir+'HighCSI_hypervol.csv'
high_df=high_df.sort_values(by=['normalized_hv'], ascending=False)
high_df.to_csv(hvFileName, index=False)
print(high_df)
print("High CSI Hyper Volume data saved in "+hvFileName)


print("\n\n----------------------------------------------------------------")
print("------------Range for Largest LOW CSI clusters------------------")
low_df = pd.DataFrame(columns = ['c_index','normalized_hv','hv', 'samplePoints'])
for i in range(len(lowCSIList)):
     leafID=lowCSIList[i][0] #ClusterID are zero-indexed and LeafID are 1 indexed
     samplePoints=lowCSIList[i][1]
     range_df=getRange(leafID)
     volScaled,vol=hyperVolume(range_df)
     #print("index {} Hypervolume={} [{}]".format(str(i), str(volScaled),str(vol)))
     low_df = low_df.append({'c_index':leafID, 'normalized_hv':volScaled, 'hv':vol, 'samplePoints':samplePoints}, ignore_index=True)

     range_df.to_csv(outputFolder+'/LowCSIRange_{}.csv'.format(str(leafID)), index=False)

     if samplePoints > 0:
          data_df = pd.DataFrame(columns = X_train.columns.tolist())
          leafIndexList=leafIndex.tolist()
          for sampleIndex in  range(len(leafIndex)):
               if leafIndex[sampleIndex] == leafID:
                    data_df=data_df.append(X_train.iloc[[sampleIndex]])
                    #print(X_train.iloc[[sampleIndex]])
          data_df.to_csv(outputFolder+'/LowCSIData_{}.csv'.format(str(leafID)), index=False)


hvFileName=communityDir+'lowCSI_hypervol.csv'
low_df=low_df.sort_values(by=['normalized_hv'], ascending=False)
low_df.to_csv(hvFileName, index=False)
print(low_df)
print("Low CSI Hyper Volume data saved in "+hvFileName)
print("\n\n All High and Low samples data available in "+outputFolder)
