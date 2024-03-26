from __future__ import print_function

import math
import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
from importlib import import_module


class DyMMMODESolver(object):
    """
    ODE Solver for DyMMM

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
        """
        A solver for Ordinary Differential Equations (ODE) specifically designed for DyMMMCommunity models. 
        It integrates the ODE system over a time span using specified initial conditions and a solver method.

        Attributes:
            community (object): An instance of DyMMMCommunity or similar, providing the initial conditions and the system of equations.

        Methods:
            __init__(self, community): Initializes the DyMMMODESolver with a specific community model.
            run(self, tspan, solver, y_init=None): Solves the ODE system for the given time span and solver.
            infeasible_event(self, t, y): An event function that stops integration when a certain condition is met.

        """
        t=None
        y=None
        if y_init is None:
            y_init=self.community.initialConditions
        if solver == 'BDF' or solver == 'Radau' or solver =='LSODA':

            sol = solve_ivp(lambda t, y: self.community.solve(t,y)
                            ,tspan, y_init, solver
                            #,events=[self.infeasible_event, self.steadystate_event]
                            ,events=[self.infeasible_event]
                            ,dense_output=True
                            #,min_step=0.1
                            ,rtol=1e-2
                            ,atol=1e-3)
                
            # if sol.status == 1:
            #     print('Success, termination event occured.')
            # if sol.status == 0:
            #     print('Success, t end reached.')

            t = sol.t
            y = sol.y.transpose()
        return t,y,sol.status 


    def infeasible_event(self, t, y):
        """
        An event function that stops the integration if the glucose concentration falls below a threshold.

        Args:
            t (float): The current time point in the integration.
            y (array): The current solution.

        Returns:
            float: The value that triggers the event when it reaches zero.
        """
        #print("Event evaulated---------------------------------------------------------{}".format(str(y)))
        return y[self.community.glucoseStateIndex] - self.infeasible_event.epsilon  #position of Glucose
    infeasible_event.epsilon = 1e-3
    #infeasible_event.direction = 0
    infeasible_event.terminal = True

if __name__ == '__main__':
    solver = DyMMMODESolver('DyMMMCommunity')
