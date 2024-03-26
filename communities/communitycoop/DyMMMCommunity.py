from __future__ import print_function

import os
import sys #system specific parameters and functions(*)
import json
import numpy
import pandas as pd
from importlib import import_module

class DyMMMCommunity(object):
    #list of species models
    species=[]
    #dictionary of states
    y={}
    #dictionary of derivatives
    dy={}
    returned_dy=None
    t=0
    last_t=0
    solverStatus=0
    fluxDf0=None
    fluxDf1=None
    
    # list of species in community
    speciesNameList=['iAF1260_Histidine_user','iAF1260_Tryptophan_user']

    #list of metabolites
    mets=['EX_glc__D_e','EX_his__L_e', 'EX_trp__L_e', 'TRPAS2', 'HISTD']	#metabolites tracked by DMMM

    # states in community
    statesList = ['Vol', 'biomass1', 'biomass2', 'EX_glc__D_e','EX_his__L_e'
                 ,'EX_trp__L_e', 'A1', 'A2', 'R1', 'R2', 'G1', 'G2'] #A1 and A2 are mislabelled. 
    
    initialConditions=numpy.array([1,0.01,0.01,5000,0.001,0.001,0.0005,0.0005,0.0001,0.0001,0.0001,0.0001])

    def initializeState(self):
           self.initialConditions=numpy.array([1,0.01,0.01,5000,0.001,0.001,0.0005,0.0005,0.0001,0.0001,0.0001,0.0001])

    #list of community parameters
    params = {
        'Fin':0.0,      #dilution into the reactor
        'Fout':0.0,      #dilution out of the reactor
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
        'Sfeed1': 0.0,
        'Sfeed2':0.0,
        'Sfeed3':0.0,
        'vmaxhis': 0.083,
        'vmaxtrp': 0.0498,
    }

    def __init__(self, communityName, communityDir):
        self.communityName=communityName      
        self.instantiateModels(communityName, communityDir)  #initiate models 
        if(len(self.initialConditions)!=len(self.statesList)): 
            sys.exit('\n\nERROR in number of initial conditions of community\n\n') #if the number of initial conditions and states don't match give error
        self.glucoseStateIndex=self.statesList.index("EX_glc__D_e")
        self.fluxDf0=pd.DataFrame()
        self.fluxDf1=pd.DataFrame() 

	#create all models in the community
    def instantiateModels(self, communityName, communityDir):
        iAF1260_Histidine_user = import_module('{}.iAF1260_Histidine_user'.format(communityName)).iAF1260_Histidine_user
        iAF1260_Tryptophan_user = import_module('{}.iAF1260_Tryptophan_user'.format(communityName)).iAF1260_Tryptophan_user
        self.species.append(iAF1260_Histidine_user(self, communityDir+"/models")) #append species list 
        self.species.append(iAF1260_Tryptophan_user(self, communityDir+"/models"))

    def setParam(self, name, value):   
            self.params[name]=value #assign value to names in list of parameters 

    # called by ODE solver to solve community at time t
    def solve(self, t, y):
    
        self.t = t
        #print(t)
                
        #print(y[self.glucoseStateIndex])

        #set states received from solver
        for i,name in enumerate(self.statesList):  #*
            self.y[name]=y[i]
            #if name == 'EX_his__L_e' or name == 'EX_trp__L_e':
            #    continue
            if self.y[name] < 0:
                self.y[name]=0 

        if(y[self.glucoseStateIndex]> 0):
            #update uptakes for each model
            for speciesModel in self.species:
                speciesModel.updateUptakes()

            #solve each model
            self.solverStatus=0
            for speciesModel in self.species:
                status = speciesModel.solve(t)
                #print("Species status "+str(status))
                #if(status == 'infeasible'):
                #    sys.exit('\n\n===>>>ERROR at time {} in {} model InitialValues or Constraints setup in UpdateUptakes\n\n'.format(t, speciesModel.__class__.__name__)) 
                if(self.solverStatus==0):
                    self.solverStatus=status
        else:
            self.solverStatus=1
            
        if(self.solverStatus!=0):
            #print("solution bypassed at "+str(y))
            dy = numpy.zeros(len(self.initialConditions))
            #dy = numpy.full(len(self.initialConditions), 1e15)
            return dy
        
        self.updateEnvironment()

        self.returned_dy=self.getDerivatives()
        self.last_t=t #setting the last time point as the new time point 
        return self.returned_dy

    def updateEnvironment(self):
        #Fin = -vs(1,1)*X(1)*V/(Sfeed(1)- S(1))
        return

    def getDerivatives(self):
        self.calculateDerivates()
        #dy = numpy.zeros(numpy.shape(len(self.dy)))
        dy = numpy.zeros(len(self.initialConditions)) #solver needs numpy. Converts dictionary to numpy. 
        for i, name in enumerate(self.statesList):
            dy[i]=self.dy[name]
        return dy


    def calculateDerivates(self):

        #if S[-1][j] == 0:
        #    self.params['kd'] = 0.081 #per hour
        #else: 
        #    self.params['kd'] = 0             
        
        p=self.params
        sp=self.species
        y=self.y.copy() 

        sp[0].F.update(t=self.t)
        sp[1].F.update(t=self.t)
        
        self.fluxDf0 = self.fluxDf0.append(sp[0].F, ignore_index=True)
        self.fluxDf1 = self.fluxDf1.append(sp[1].F, ignore_index=True)
        

        #dV/dt [L/hr]
        self.dy['Vol'] = p['Fin'] - p['Fout'] 

        #dX/dt [g/L/hr]
        #self.dy['biomass1'] =  sp[0].F['BIOMASS_Ec_iML1515_core_75p37M']*y['biomass1'] + p['Fin']/y['Vol']*(p['Xfeed1'] - y['biomass1']) #confirm with the original code
        #self.dy['biomass2'] =  sp[1].F['BIOMASS_Ec_iML1515_core_75p37M']*y['biomass2'] + p['Fin']/y['Vol']*(p['Xfeed2'] - y['biomass2']) 

        K = 0.7
        kd = 0.0#0.018
        
        self.dy['biomass1'] = sp[0].F['BIOMASS_Ec_iML1515_core_75p37M']*(((K-y['biomass1'])/K))*y['biomass1']-kd*y['biomass1'] + p['Fin']/y['Vol']*(p['Xfeed1'] - y['biomass1'])
        self.dy['biomass2'] = sp[1].F['BIOMASS_Ec_iML1515_core_75p37M']*(((K-y['biomass2'])/K))*y['biomass2']-kd*y['biomass2'] + p['Fin']/y['Vol']*(p['Xfeed2'] - y['biomass2'])
        
        #dS/dt [mmol/L/hr]
        self.dy['EX_glc__D_e'] =  (y['biomass1'] * sp[0].F['EX_glc__D_e']+ y['biomass2'] * sp[1].F['EX_glc__D_e']) + p['Fin']/y['Vol']*(p['Sfeed1']-y['EX_glc__D_e'])
  
        self.dy['EX_his__L_e'] =  (y['biomass1'] * sp[0].F['EX_his__L_e']+ y['biomass2'] * sp[1].F['EX_his__L_e']) + p['Fin']/y['Vol']*(p['Sfeed2']-y['EX_his__L_e'])

        self.dy['EX_trp__L_e'] =  (y['biomass1'] * sp[0].F['EX_trp__L_e']+ y['biomass2'] * sp[1].F['EX_trp__L_e']) + p['Fin']/y['Vol']*(p['Sfeed3']-y['EX_trp__L_e'])   
        
        #interaction equations
        
        self.dy['A1'] = p['va1']*y['biomass1'] - p['gammaa1']*y['A1']  #A1
        
        self.dy['A2'] =  p['va2']*y['biomass2'] - p['gammaa2']*y['A2'] 

        # if y['R1'] < 0:
        #     self.dy['R1']=0.0
        # else:
        self.dy['R1'] = p['p1'] *y['A1']**2*p['luxR']**2 - p['gammaa3']*y['R1'] - sp[1].F['BIOMASS_Ec_iML1515_core_75p37M']*y['R1']
        
        # if y['R2'] < 0:
        #     self.dy['R2']=0.0
        # else:
        self.dy['R2'] = p['p2'] *y['A2']**2*p['lasR']**2 - p['gammaa4']*y['R2'] - sp[0].F['BIOMASS_Ec_iML1515_core_75p37M']*y['R2']
        
        #print(" {}  {} ".format(str(sp[0].F['BIOMASS_Ec_iML1515_core_75p37M']), str(sp[0].F['BIOMASS_Ec_iML1515_core_75p37M'])))

        # if y['G1'] < 0:
        #     self.dy['G1']=0.0
        # else:
            #print(p['n'])
            #print(y['R2'])
            #print(y['R2']**p['n'])
            #print((p['theta']**p['n'] + y['R2']**p['n']))
        self.dy['G1'] = ((p['alpha1'] * (1-numpy.exp(-sp[0].F['BIOMASS_Ec_iML1515_core_75p37M']/0.33)) * y['R2']**p['n']) /(p['theta']**p['n'] + y['R2']**p['n'])) - p['gammaa5']*y['G1'] - sp[0].F['BIOMASS_Ec_iML1515_core_75p37M']*y['G1']
        
        #if self.dy['G1']+y['G1'] < 0:
        #    self.dy['G1'] = -y['G1'] + 1e-70
        
        #if y['G2'] < 0:
        #    self.dy['G2']=0.0
        #else:
        self.dy['G2'] = ((p['alpha2'] * (1-numpy.exp(-sp[1].F['BIOMASS_Ec_iML1515_core_75p37M']/0.33)) * y['R1']**p['n']) /(p['beta']**p['n'] + y['R1']**p['n'])) - p['gammaa6']*y['G2'] - sp[1].F['BIOMASS_Ec_iML1515_core_75p37M']*y['G2']
        
        #if self.dy['G2']+y['G2'] < 0:
        #    self.dy['G2'] = -y['G2'] + 1e-70

        #print(self.dy)
        #print(self.y)
        return self.dy

    def getFluxes(self):
        return

if __name__ == '__main__':
    communityDir='.'
    community = DyMMMCommunity(communityDir)
    print(community.species)


    