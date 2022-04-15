from __future__ import print_function

import math
import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
from importlib import import_module


class DyMMMODESolver(object):
    """
    ODE Solver. 

    Attributes
    ----------

    Methods
    -------
    """    

    community=None

    def __init__(self, community):
        #self.DyMMMCommunity = import_module('{}.DyMMMCommunity'.format(communityDir))     
        self.community=community

    def run(self, tspan, solver, y_init=None):
        t=None
        y=None
        if y_init is None:
            y_init=self.community.initialConditions
        if solver == 'BDF' or solver == 'Radau' or solver =='LSODA':

            sol = solve_ivp(lambda t, y: self.community.solve(t,y)
                            ,tspan, y_init, solver
                            #,events=[self.infeasible_event, self.steadystate_event]
                            ,events=[self.infeasible_event]
                            ,dense_output=True)
                
            # if sol.status == 1:
            #     print('Success, termination event occured.')
            # if sol.status == 0:
            #     print('Success, t end reached.')

            t = sol.t
            y = sol.y.transpose()
        return t,y,sol.status 


    def infeasible_event(self, t, y):
        #print("Event evaulated---------------------------------------------------------{}".format(str(y)))
        return y[self.community.glucoseStateIndex] - self.infeasible_event.epsilon  #position of Glucose
    infeasible_event.epsilon = 1e-3
    #infeasible_event.direction = 0
    infeasible_event.terminal = True

if __name__ == '__main__':
    solver = DyMMMODESolver('DyMMMCommunity')
