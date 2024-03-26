from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import DyMMMDataPlot
import gzip


from importlib import import_module
import DyMMMSettings as settings
from DyMMMODESolver import DyMMMODESolver
from scipy import integrate


params1 = {
'Fin':0.03,
'Fout':0.03,
'va1':16.99983369,
'va2':16.99989222,
#'gammaa1':8.319948064,
#'gammaa2':1.84295429,
'gammaa3':139.9998033,
'gammaa4':0.014338118,
'gammaa5':119.9953557,
'gammaa6':119.999339,
'p1':830132.0502,
'p2':830377.2225,
'luxR':0.000271426,
'lasR':0.000250018,
'alpha1':0.060901481,
'alpha2':0.061882525,
'k1':99.99875108,
'k2':1.007260049,
'n':4.99990394,
'theta':1.00E-04,
'beta':3.76E-06,
'Fin':0.499998476
}
params2 = {
        'Fin':0.55,      #dilution into the reactor
        'Fout':0.55,      #dilution out of the reactor
        'va1':1.7,
        'va2':1.7,  #AHL production rate constant in mm L^-1 hr^-1 1.7
        'gammaa1': 0.6, #0.6, #AHL decay constant in hr^-1
        'gammaa2':0.6,
        'gammaa3': 1.4, #1.4, 
        'gammaa4':1.4,
        'gammaa5':1.2,
        'gammaa6':1.2,
        'p1':8300000, #AHL/LasR dimerization constant in mM^-3 hr^-1
        'p2': 8300000, #8300000, 
        'luxR': 0.0005, #0.0005, #mM
        'lasR':0.0005,
        'kd':0.0,
        'alpha1':0.06,
        'alpha2': 0.06,#0.06, #protein synthesis rate mM hr^-1 0.06
        'Xfeed1':0.0,
        'Xfeed2':0.0,
        'k1' : 9.96, #9.96, #8.4 - used in model updateUptakes
        'k2' : 16.6, #16.6, #8.4 - used in model updateUptakes        
        'n':2, #transcription factor cooperativity 
        'theta': 0.00001, #LuxR/AHL activation coefficient mM^2. LacI repression = 0.0008 mM. CI repression = 0.000008 mM. 
        'beta':0.00001, 
        'Sfeed1': 100.0,
        'Sfeed2':0.0,
        'Sfeed3':0.0,
        'vmaxhis': 0.083,
        'vmaxtrp': 0.0498,
    }

params = {
        'va1':4.189362581240542,
        'va2':2.927597307035691,  #AHL production rate constant in mm L^-1 hr^-1 1.7
        'gammaa3':120.66777979886093, #1.4,
        'gammaa4':79.20330152939339,
        'gammaa5':84.12634139428867,
        'gammaa6':43.10595565447185,
        'p1':4680513.201764763, #AHL/LasR dimerization constant in mM^-3 hr^-1
        'p2':73174940.2963288, #8300000,
        'luxR': 0.00027452843538091926, #0.0005, #mM
        'lasR':0.00033329220858827865,
        'alpha1':141.3091638120059,
        'alpha2':489.21431642756636,#0.06, #protein synthesis rate mM hr^-1 0.06
        'k1' : 95.60828576820556, #9.96, #8.4 - used in model updateUptakes
        'k2' : 97.4729383775986, #16.6, #8.4 - used in model updateUptakes
        'n':1.5504766342586265, #transcription factor cooperativity
        'theta':1.179402182307254e-06, #LuxR/AHL activation coefficient mM^2. LacI repression = 0.0008 mM. CI repression = 0.000008 mM.
        'beta':4.402650302143001e-05
    }

def computeCSI(self, df):
    """
    Computes the Community Stability Index (CSI) based on a provided DataFrame.
    The function calculates the CSI using a formula that takes into account the
    proportions of biomass of different species in the community.

    Parameters:
        df (DataFrame): A DataFrame containing the time series data of the system.

    Returns:
        float: The computed CSI value.
    """
    CSI=0
    lastBiomass1=df['biomass1'].iloc[-1]
    lastBiomass2=df['biomass2'].iloc[-1]
    if (lastBiomass1 > 1e-3 or lastBiomass2 > 1e-3):
        p1=lastBiomass1/(lastBiomass1+lastBiomass2)
        p2=lastBiomass2/(lastBiomass1+lastBiomass2)
        Sp1=p1*np.log(p1)
        Sp2= p2*np.log(p2)
        CSI=  (Sp1+Sp2)/(np.log(2)) * (-1)
    #print(CSI)
    return CSI


def isSteadyState(df, colName):
    time2=df['time'].iloc[-1]
    if(time2 < 3):
        return False
    time1=time2-1
    time0=time1-1
    row0=df.loc[(df['time'] <= time0)]
    row1=df.loc[(df['time'] <= time1)]  
    row2=df.loc[(df['time'] <= time2)]  
    value0=row0[colName].iloc[-1]
    value1=row1[colName].iloc[-1]
    value2=row2[colName].iloc[-1]
    currentDerivative=(value2-value1)/(time2-time1)
    prevDerivative=(value1-value0)/(time1-time0)    
    #error1=abs(currentDerivative-prevDerivative)
    error1=max(abs(currentDerivative), abs(prevDerivative))
    error2=abs(value2-value0)
    # error1=abs(currentDerivative-prevDerivative)
    # error2=abs(value2-value0)/max(value1, value0) 
    error=max(error1, error2)
    print("----------------")
    print(error1)
    print(error2)    
    print(error)
    return error < 1e-6


if __name__ == '__main__':

    print("started")

    # df=pd.read_csv("data/SteadyState_test.csv")
    # print(df)
    # print(isSteadyState(df,'biomass2'))
    # #DyMMMDataPlot.plot1(df,"PRed")    
    # exit(0)

    communitiesDir=settings.simSettings["communitiesDir"]
    communityName=settings.simSettings["communityName"]

    solverName=settings.simSettings["solverName"]
    sys.path.append(communitiesDir)
    stopTime=settings.simSettings['stopTime']
    communityDir=communitiesDir+"/"+communityName
    DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(communityName)).DyMMMCommunity
    community=DyMMMCommunity(communityName, communityDir)

    for paramName in params:
        community.setParam(paramName,params[paramName])

    solver=DyMMMODESolver(community)


    tStart=0
    tMax=50
    stepSize=1
    # sampleRate = 100
    # frequency = 1
    # length = 5

    # t_perturb = np.linspace(0, length, sampleRate * length)  
    # y_perturb = 1e-6 * np.sin(frequency *  q2 * np.pi * t)  

    tEnd=stepSize
    t=None
    y=None
    init_values=None
    while tEnd < tMax:
        tspan = [tStart, tEnd]
        print(tspan)
        t_temp,y_temp, status=solver.run(tspan,'BDF', init_values)
        if t is None:
            t=t_temp
            y=y_temp
        else:
            t=np.append(t, t_temp[1:],axis = 0) 
            y=np.append(y, y_temp[1:],axis = 0) 
            print("y count {}".format(str(y.shape)))
        init_values=y[-1]
        df=pd.DataFrame(data=y,
                        index=t,
                        columns=community.statesList)
        df.index.name = 'time'
        df.reset_index(level=0, inplace=True)
        ss1=isSteadyState(df,'biomass1')
        ss2=isSteadyState(df,'biomass2')
        #ss3=isSteadyState(df,'biomass3')
        print("Steady State index {} {}".format(str(ss1),str(ss2)))
        tStart+=stepSize
        tEnd+=stepSize
        if(ss1 and ss2):
            break
        if status==1:
            print("Exit on status")
            break
        
    dataFrame=pd.DataFrame(data=y,
                    index=t,
                    columns=community.statesList)

    dataFrame.index.name = 'time'
    dataFrame.reset_index(level=0, inplace=True)

    dataFrame['M1'] = dataFrame['biomass1'].diff()/dataFrame['time'].diff()
    dataFrame['M2'] = dataFrame['biomass2'].diff()/dataFrame['time'].diff()
    #dataFrame['M3'] = dataFrame['biomass3'].diff()/dataFrame['time1'].diff()
    dataFrame.fillna(0, inplace=True)

    print(isSteadyState(dataFrame,'biomass1'))
    print(isSteadyState(dataFrame,'biomass2'))
    #print(isSteadyState(dataFrame,'biomass3'))

    CSI=computeCSI(dataFrame)
    print("=====================================================")
    print("CSI = "+str(CSI))
    print("=====================================================")

    outFile="data/"+communityName
    dataFrame.to_csv(outFile+".csv", sep=',')

    community.fluxDf0.to_csv(outFile+"_HUSER.csv", sep=',', index=False, compression='gzip')
    community.fluxDf1.to_csv(outFile+"_TUSER.csv", sep=',', index=False, compression='gzip')
    #community.fluxDf2.to_csv(outFile+"_IUSER.csv", sep=',', index=False, compression='gzip')

    DyMMMDataPlot.plot1(None, communityName, outFile+".csv")
   