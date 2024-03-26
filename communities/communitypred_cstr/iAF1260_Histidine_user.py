from __future__ import print_function

import os
import sys
import json
import numpy
import cobra

class iAF1260_Histidine_user(object):
 
    cobra_solver = 'glpk'

    community=None
    model=None
    model_rnx=None
    solverStatus=0

    reactions=['BIOMASS_Ec_iML1515_core_75p37M','EX_glc__D_e','EX_his__L_e', 'EX_trp__L_e','HISTD', 'TRPS1','TRPS2','TRPAS2', 'TRPt2rpp','MTRPOX','HISt2rpp']

    #flux
    #fluxList = ['biomass','EX_glc__D_e','EX_his__L_e', 'EX_trp__L_e']
    F={}

    def __init__(self, community, modelDir):
        self.community=community
        self.model = cobra.io.load_json_model("{}/iML1515.json".format(modelDir))

        #setup Solver for this model
        self.model.solver = self.cobra_solver
        self.model.solver.configuration.timeout=10
        
        #Setup Reactions
        self.model_rnx = {}

        for reactionName in self.reactions:
            self.model_rnx[reactionName] = self.model.reactions.get_by_id(reactionName)

        self.model_rnx['TRPS1'].lower_bound = 0
        self.model_rnx['TRPS1'].upper_bound = 0
        self.model_rnx['TRPt2rpp'].bounds = (-1000,1000)        
        self.model_rnx['MTRPOX'].lower_bound = 0
        self.model_rnx['MTRPOX'].upper_bound = 0
        self.model_rnx['TRPAS2'].lower_bound = 0
        self.model_rnx['TRPAS2'].upper_bound = 0
        self.model_rnx['TRPS2'].lower_bound = 0
        self.model_rnx['TRPS2'].upper_bound = 0

    def solve(self, t):
        self.solverStatus=0
        sol=None
        try:
            sol = self.model.optimize()
        except Exception as e:
            print(e)
            print('model {} is NOT optimal'.format(type(self).__name__))
            self.solverStatus=1
            #self.model.summary()
        #if sol.status == 'infeasible': #to be used if solver is not working for first solve
        #self.printConstraints()
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

        #tryptophan production
        k=p['k1']
        Vmax_tp = -k*y['G1']
        if Vmax_tp <= -0.0498*2:
            Vmax_tp = -0.0498*2 #0.253  #Assume Ks on glucose same as E. coli  #k1 = 8.4, k2 = 13.8 
        else:
            Vmax_tp = -k*y['G1']
        self.model_rnx['TRPAS2'].lower_bound = Vmax_tp

        # tryptophan import 
        C = y['EX_trp__L_e']/y['Vol']
        vmaxtrp = p['vmaxtrp']
        if C >= 0:
            Vmax_ti = -vmaxtrp*C/(C + 0.001)  #Assume Ks on glucose same as E. coli
        else:
            Vmax_ti = 0
        self.model_rnx['EX_trp__L_e'].lower_bound = Vmax_ti
        self.model_rnx['EX_trp__L_e'].upper_bound = 0

    def printConstraints(self):
        for name,reaction in self.model_rnx.items():
            print('Reaction {} lower_bound {} upper_bound {}'.format(name, str(reaction.lower_bound), str(reaction.upper_bound)), flush=True)
        print('Last flux values')
        for name,value in self.F.items():
            print('Flux {} value {}'.format(name, str(value), flush=True)) #*

if __name__ == '__main__':
    None #*
