
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from cycler import cycler
from itertools import cycle
import seaborn as sns
import DyMMMSettings as settings




lines = ["-","--","-.",":"]
linecycler = cycle(lines)
color=['r', 'g', 'b', 'y']
colorcycler = cycle(color)

def plot1(dataFrame, titleText, filePath=None):

    if filePath is not None:
        dataFrame=pd.read_csv(filePath, index_col='time')
    print(dataFrame)
    x = dataFrame.index
    Biomass1 = dataFrame['biomass1']
    Biomass2 = dataFrame['biomass2']
    Glucose  = dataFrame['EX_glc__D_e']
    G1  = dataFrame['G1']
    G2  = dataFrame['G2']
    R1  = dataFrame['R1']
    R2 = dataFrame['R2']
    DIFF = (Biomass1 - Biomass2)

    fig=plt.figure()
    gs = gridspec.GridSpec(8, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1]) 


    # first subplot
    fig.suptitle(titleText)
    ax0 = plt.subplot(gs[0])
    line0, = ax0.plot(x, Biomass1, color='r')

    # second subplot
    ax1 = plt.subplot(gs[1], sharex = ax0)
    line1, = ax1.plot(x, Biomass2, color='b')
    plt.setp(ax0.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

   # third subplot
    ax2 = plt.subplot(gs[2], sharex = ax1)
    line2, = ax2.plot(x, Glucose, color='g')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

   # fourth subplot
    ax3 = plt.subplot(gs[3], sharex = ax2)
    line3, = ax3.plot(x, G1, color='r', linestyle='dashed')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax3.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

   # fifth subplot
    ax4 = plt.subplot(gs[4], sharex = ax3)
    line4, = ax4.plot(x, G2, color='b', linestyle='dashed')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax4.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)


   # 6th subplot
    ax5 = plt.subplot(gs[5], sharex = ax3)
    line5, = ax5.plot(x, R1, color='r', linestyle='dotted')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax5.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

   # 7th subplot
    ax6 = plt.subplot(gs[6], sharex = ax3)
    line6, = ax6.plot(x, R2, color='b', linestyle='dotted')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax6.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)    

   # 8th subplot
    ax7 = plt.subplot(gs[7], sharex = ax3)
    line7, = ax7.plot(x, DIFF, color='g', linestyle='dotted')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax7.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)    

    ax0.legend((line0, line1, line2, line3, line4, line5, line6, line7), ('Biomass1', 'Biomass2'
                , 'Glucose', 'G1', 'G2', 'R1', 'R2','BiomassDiff'), loc='lower left')

    # remove vertical gap between subplots

    plt.subplots_adjust(hspace=.1)
    plt.show()


def plot2(titleText, filePath=None):
    
    df=pd.read_csv(filePath, compression='gzip')
    colCount=df.shape[1]
    fig=plt.figure()
    fig.suptitle(titleText)

    gs = gridspec.GridSpec(colCount, 1) 

    x = df.index
    nameList=[]
    lineList=[]

    ax0 = plt.subplot(gs[0])
    line0, = ax0.plot(x, df.iloc[:,0], color='r')
    lineList.append(line0)
    nameList.append(df.columns[0])

    for index in range(1,colCount):
        ax = plt.subplot(gs[index], sharex = ax0)
        line, = ax.plot(x, df.iloc[:,index],linestyle=next(linecycler), color=next(colorcycler))
        plt.setp(ax0.get_xticklabels(), visible=False)
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)         
        lineList.append(line)
        nameList.append(df.columns[index])
        dummy=next(colorcycler)

    ax0.legend(lineList,nameList,loc='lower left')
    plt.subplots_adjust(hspace=.1)
    plt.show()


def plotComplexVector(z):
    X = [x.real for x in z]
    Y = [x.imag for x in z]
    plt.scatter(X,Y, color='red')
    plt.show()

def showViolinPlot2(CSIDF, columnNames, Ylabel):
    result=columnNames[0]
    if(len(columnNames) > 2):
        CSIDF['result']=CSIDF[columnNames[0]] - CSIDF[columnNames[1]]
        result='result'
    ax = sns.violinplot(x="community", y='CSI', data=CSIDF)
    ax.set_ylabel(Ylabel)
    plt.show()


def plotViolin(columnName='CSI', Ylabel='Diversity', titletext='diversity'):
    communitycoopCSI=settings.simSettings["communitiesDir"]+"/communitycoop_cstr_RESULT.csv"
    communitycompCSI=settings.simSettings["communitiesDir"]+"/communitycomp_cstr_RESULT.csv"
    communitypredCSI=settings.simSettings["communitiesDir"]+"/communitypred_cstr_RESULT.csv"
    
    communitycoopCSIDF=pd.read_csv(communitycoopCSI)
    communitycoopCSIDF['community'] = "COOP"
    communitycompCSIDF=pd.read_csv(communitycompCSI)
    communitycompCSIDF['community'] = "COMP"
    communitypredCSIDF=pd.read_csv(communitypredCSI)
    communitypredCSIDF['community'] = "PRED"

    #communitycoopCSIDF=communitycoopCSIDF.loc[communitycoopCSIDF['CSI'] > 0.9]
    #communitycompCSIDF=communitycompCSIDF.loc[communitycompCSIDF['CSI'] > 0.9]
    #communitypredCSIDF=communitypredCSIDF.loc[communitypredCSIDF['CSI'] > 0.9] 
    #communitycoopCSIDF=communitycoopCSIDF.loc[communitycoopCSIDF['CSI'] < 0.2]
    #communitycompCSIDF=communitycompCSIDF.loc[communitycompCSIDF['CSI'] < 0.2]
    #communitypredCSIDF=communitypredCSIDF.loc[communitypredCSIDF['CSI'] < 0.2] 

    columnNames = [columnName]

    columnNames.append('community')
    #columnNames.append('EX_glc__D_e')
    
    df_1=communitycoopCSIDF[columnNames]
    print(df_1.shape)
    df_2=communitycompCSIDF[columnNames]
    print(df_2.shape)
    df_3=communitypredCSIDF[columnNames]
    print(df_3.shape)

    combinedCSIDF=df_1
    combinedCSIDF=combinedCSIDF.append(df_2, ignore_index=True)
    combinedCSIDF=combinedCSIDF.append(df_3, ignore_index=True)

    # print(combinedCSIDF.loc[combinedCSIDF['biomass2_SS'] < 0])
    # combinedCSIDF=combinedCSIDF.loc[combinedCSIDF['biomass2_SS'] < 1]
    # combinedCSIDF=combinedCSIDF.loc[combinedCSIDF['biomass2_SS'] >= 0]


    ax = sns.violinplot(x="community", y=columnName, data=combinedCSIDF)
    ax.set_ylabel(Ylabel)
    plt.title(titletext)
    plt.show()



def plotViolin_1(columnName='CSI', Ylabel='Diversity'):
    communitycoopCSI=settings.simSettings["communitiesDir"]+"/communitycoop_cstr_RESULT.csv"
    communitycompCSI=settings.simSettings["communitiesDir"]+"/communitycomp_cstr_RESULT.csv"
    communitypredCSI=settings.simSettings["communitiesDir"]+"/communitypred_cstr_RESULT.csv"
    communitycoopCSIDF=pd.read_csv(communitycoopCSI)
    communitycoopCSIDF['community'] = "COOP"
    communitycompCSIDF=pd.read_csv(communitycompCSI)
    communitycompCSIDF['community'] = "COMP"
    communitypredCSIDF=pd.read_csv(communitypredCSI)
    communitypredCSIDF['community'] = "PRED"

    #communitycoopCSIDF=communitycoopCSIDF.loc[communitycoopCSIDF['CSI'] > 0.9]
    #communitycompCSIDF=communitycompCSIDF.loc[communitycompCSIDF['CSI'] > 0.9]
    #communitypredCSIDF=communitypredCSIDF.loc[communitypredCSIDF['CSI'] > 0.9]


    columnNames = [columnName]

    columnNames.append('community')
    columnNames.append('EX_glc__D_e')
    
    df_1=communitycoopCSIDF[columnNames]
    print(df_1.shape)
    df_2=communitycompCSIDF[columnNames]
    print(df_2.shape)
    df_3=communitypredCSIDF[columnNames]
    print(df_3.shape)


    df_1['EX_glc__D_e'] = 5000 - df_1['EX_glc__D_e']
    df_2['EX_glc__D_e'] = 5000 - df_2['EX_glc__D_e']
    df_3['EX_glc__D_e'] = 5000 - df_3['EX_glc__D_e']

    df_1['biomassDiff']=communitycoopCSIDF['biomass1']-communitycoopCSIDF['biomass2']
    print(df_1.shape)
    df_2['biomassDiff']=communitycompCSIDF['biomass1']-communitycompCSIDF['biomass2']
    print(df_2.shape)
    df_3['biomassDiff']=communitypredCSIDF['biomass1']-communitypredCSIDF['biomass2']
    print(df_3.shape)


    combinedCSIDF=df_1
    combinedCSIDF=combinedCSIDF.append(df_2, ignore_index=True)
    combinedCSIDF=combinedCSIDF.append(df_3, ignore_index=True)

    # print(combinedCSIDF.loc[combinedCSIDF['biomass2_SS'] < 0])
    # combinedCSIDF=combinedCSIDF.loc[combinedCSIDF['biomass2_SS'] < 1]
    # combinedCSIDF=combinedCSIDF.loc[combinedCSIDF['biomass2_SS'] >= 0]


    ax = sns.violinplot(x="community", y=columnName, data=combinedCSIDF)
    ax.set_ylabel(Ylabel)
    plt.show()


def showScatterMatrix(CSIDF, rangeDF=None):


    print(CSIDF.shape)
    high=CSIDF.loc[CSIDF['CSI'] > 0.9]
    print(high.shape)


    #CSIDF=pd.read_csv(inputFile)

    # CSIDF_HIGH=CSIDF.loc[(CSIDF['CSI'] > 0.9)]
    # CSIDF_HIGH = CSIDF_HIGH.drop('CSI', axis=1)
    # CSIDF_LOW=CSIDF.loc[(CSIDF['CSI'] < 0.9)]
    # CSIDF_LOW = CSIDF_LOW.drop('CSI', axis=1)

    # axes=pd.plotting.scatter_matrix(CSIDF_HIGH, alpha=0.2)

    # for i in range(np.shape(axes)[0]):
    #     for j in range(np.shape(axes)[1]):
    #         if i < j:
    #             axes[i,j].set_visible(False)

    # plt.show()

    if rangeDF is not None:
        for index, row in rangeDF.iterrows():
            #CSIDF=CSIDF.loc[(CSIDF[row['Parameter']] > row['MinValue']) & (CSIDF[row['Parameter']] < row['MaxValue']) ]
            CSIDF=CSIDF.loc[(CSIDF[row['Parameter']] > row['MinValue'])]
            CSIDF=CSIDF.loc[(CSIDF[row['Parameter']] < row['MaxValue'])]

    print(CSIDF)
    CSIDF.loc[CSIDF['CSI'] >= 0.9, 'Diversity'] = 'High'
    CSIDF.loc[(CSIDF['CSI'] >= 0.5) & (CSIDF['CSI'] < 0.9), 'Diversity'] = 'Medium'
    CSIDF.loc[CSIDF['CSI'] < 0.5, 'Diversity'] = 'Low'
    CSIDF = CSIDF.drop('CSI', axis=1)
    g = sns.pairplot(CSIDF, corner=True, hue="Diversity")
    #g = sns.pairplot(CSIDF, corner=True, hue="CSIX", plot_kws=dict(marker="+", linewidth=1))
    # for ax in g.axes.flatten():
    #     # rotate x axis labels
    #     ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    #     # rotate y axis labels
    #     ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    #     # set y labels alignment
    #     ax.yaxis.get_label().set_horizontalalignment('right')
    plt.show()



def plotHyperVolume():
    communitiesDir=settings.simSettings["communitiesDir"]
    communitycoop_hv=pd.read_csv(communitiesDir+"/communitycoop_hv.csv",names=['value'])
    communitycomp_hv=pd.read_csv(communitiesDir+"/communitycomp_hv.csv",names=['value'])
    communitypred_hv=pd.read_csv(communitiesDir+"/communitypred_hv.csv",names=['value'])

    communitycoop_hv=communitycoop_hv.sort_values(by=['value'], ascending=False)[0:100].reset_index()
    print(communitycoop_hv)

    communitycomp_hv=communitycomp_hv.sort_values(by=['value'], ascending=False)[0:100].reset_index()
    print(communitycomp_hv)

    communitypred_hv=communitypred_hv.sort_values(by=['value'], ascending=False)[0:100].reset_index()
    print(communitypred_hv)

    plt.plot(communitycoop_hv['value'], color='red', label='community-coop') 
    plt.plot(communitycomp_hv['value'], color='blue', label='community-comp') 
    plt.plot(communitypred_hv['value'], color='green', label='community-pred') 
    plt.title("Top 100 Normalized Hypervolumes")
    plt.legend()
    plt.show()


def plotComplexNumbers(communitiesDir, communityName, title, index, highCSI):

    eigenFile=communitiesDir+"/"+communityName+"_eigen.txt"
    eigen=np.loadtxt(eigenFile).view(complex)
    #eigen=pd.DataFrame(data=eigen_np, columns=['Vol', 'biomass1', 'biomass2', 'EX_glc__D_e','EX_his__L_e'
    #             ,'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2'])
    
    communityCSI=settings.simSettings["communitiesDir"]+"/{}_RESULT.csv".format(communityName)
    communityCSIDF=pd.read_csv(communityCSI)

    communityCSIDF=communityCSIDF.loc[communityCSIDF['CSI'] < 0.9]
    #eigen=eigen[eigen.index.isin(communityCSIDF.index)]

    print(eigen.shape)
    if highCSI:
        eigen=np.delete(eigen,communityCSIDF.index.tolist(), axis=0)
    print(eigen.shape)

    X = [x[index].real for x in eigen]
    Y = [x[index].imag for x in eigen]
    plt.scatter(X,Y, color='red', marker=".")
    plt.title(title)
    plt.show()
    