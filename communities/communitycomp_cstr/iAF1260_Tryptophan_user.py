from __future__ import print_function

import os
import sys
import json
import numpy
import cobra

class iAF1260_Tryptophan_user(object):
    cobra_solver = 'glpk'

    community=None
    model=None
    model_rnx=None
    solverStatus=0
    

    reactions=['BIOMASS_Ec_iML1515_core_75p37M','EX_glc__D_e','EX_his__L_e', 'EX_trp__L_e', 'TRPAS2', 'HISTD','HISabcpp','HISt2rpp']
    #flux
    #fluxList = ['biomass','EX_glc__D_e','EX_his__L_e', 'EX_trp__L_e']
    F={}

    def __init__(self, community, modelDir):
        self.community=community
        self.model = cobra.io.load_json_model("{}/iML1515.json".format(modelDir))

        #setup Solver for this model
        self.model.solver = self.cobra_solver
        
        #Setup Reactions
        self.model_rnx = {}

        for reactionName in self.reactions:
            self.model_rnx[reactionName] = self.model.reactions.get_by_id(reactionName)

        self.model_rnx['HISTD'].lower_bound = 0
        self.model_rnx['HISTD'].upper_bound = 0
        self.model_rnx['HISabcpp'].lower_bound = 0
        self.model_rnx['HISabcpp'].upper_bound = 0
        self.model_rnx['HISt2rpp'].bounds = (-1000,1000)  
      

    def solve(self, t):
        self.solverStatus=0
        try:
            sol = self.model.optimize()
        except Exception as e:
            print('model {} is NOT optimal'.format(type(self).__name__))
            self.solverStatus=1
            #self.model.summary()
        #if sol.status == 'infeasible': #to be used if solver is not working for first solve
        #    self.printConstraints()
        if sol.status != 'optimal': 
            print('model {} is NOT optimal status:{} at {}'.format(type(self).__name__, sol.status, str(t)))
            self.solverStatus=1
            self.F['BIOMASS_Ec_iML1515_core_75p37M'] = 0
            for name in self.reactions:
                self.F[name]=0
            #print('species {} is dying',format(self.__class__.__name__))
        else:
            self.solverStatus=0
            #for name in self.reactions:
            #    self.F[name]=sol.fluxes[name]
            for name, value in sol.fluxes.items():
                self.F[name]=sol.fluxes[name]           
        return self.solverStatus

    def updateUptakes(self):
        # glucose
        y=self.community.y
        p=self.community.params

        C = y['EX_glc__D_e']/y['Vol']
        if  C >= 0:
            Vmax_glc = -10*C/(C + 0.015)  #Assume Ks on glucose same as E. coli
        else:
            Vmax_glc = 0
        self.model_rnx['EX_glc__D_e'].lower_bound = Vmax_glc
        
        # histidine import 
        C = y['EX_his__L_e']/y['Vol']
        vmaxhis = p['vmaxhis']
        if C >= 0:
            Vmax_hi = -vmaxhis*C/(C + 0.001)  #Assume Ks on glucose same as E. coli
        else:
            Vmax_hi = 0
        self.model_rnx['EX_his__L_e'].lower_bound = Vmax_hi
        self.model_rnx['EX_his__L_e'].upper_bound = 0

        #histidine production
        k=p['k2']
        Vmax_hp = k*y['G2']
        if Vmax_hp >= 2*0.083:
            Vmax_hp = 2*0.083  #Assume Ks on glucose same as E. coli
        else:
            Vmax_hp = k*y['G2']
        self.model_rnx['HISTD'].upper_bound = Vmax_hp
        self.model_rnx['HISTD'].lower_bound = 0

    def printConstraints(self):
        for name,reaction in self.model_rnx.items():
            print('Reaction {} lower_bound {} upper_bound {}'.format(name, str(reaction.lower_bound), str(reaction.upper_bound)), flush=True)
        for name,value in self.F.items():
            print('Flux {} value {}'.format(name, str(value), flush=True))

if __name__ == '__main__':
    None
