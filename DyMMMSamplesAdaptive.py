import os
import sys
import numpy as np
import glob
import pandas as pd
import DyMMMSettings as settings
from DyMMMMultiObjectiveProblem import DyMMMMultiObjectiveProblem
from sklearn.preprocessing import MinMaxScaler
from pymoo.optimize import minimize
from DyMMMSurrogateModel import DyMMMSurrogateModel
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination

from pymoo.visualization.scatter import Scatter

class DyMMMSamplesAdaptive:

    localWeight=0
    globalWeight=1
    problem=None
    algorithm=None
    lowerBounds=None
    upperBounds=None
    scaler=None

    def __init__(self, paramCount, lowerBounds, upperBounds, scaler, surrogate):
        self.paramCount = paramCount
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.surrogate = surrogate
        self.scaler = scaler
        self.algorithm = NSGA2(
                                pop_size=80,
                                n_offsprings=20,
                                sampling=get_sampling("real_random"),
                                crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                                mutation=get_mutation("real_pm", eta=20),
                                eliminate_duplicates=True
                                )

    def sample(self, train_X, train_y, n_last=150):
        self.problem = DyMMMMultiObjectiveProblem(  self.paramCount,
                                                    self.lowerBounds,
                                                    self.upperBounds,
                                                    train_X,
                                                    train_y,
                                                    self.scaler,
                                                    self.surrogate)
        tol=0.01
        #if None!=os.environ['DyMMM_PYMOO_tol']:                              
        #    tol=float(os.environ['DyMMM_PYMOO_tol'])
        termination = DesignSpaceToleranceTermination(tol=tol, n_last=30)
        #termination = DesignSpaceToleranceTermination(tol=tol, n_last=min(n_last,self.paramCount*15))
        #termination = get_termination("time", "00:15:00")

        # termination = MultiObjectiveSpaceToleranceTermination(tol=0.0025,
        #                                               n_last=30,
        #                                               nth_gen=5,
        #                                               n_max_gen=None,
        #                                               n_max_evals=None)
        res = minimize(self.problem, self.algorithm, termination, seed=1, save_history=True, verbose=False)
        ps = self.problem.pareto_set(use_cache=False, flatten=False)
        pf = self.problem.pareto_front(use_cache=False, flatten=False)
        return res, ps, pf

def generateTrainingDF(inputDataDir):
    files=glob.glob(inputDataDir+"/*_RESULT.csv")
    train_df = pd.DataFrame()
    n=len(files)
    for i in range(n):
        inputDataFile = inputDataDir+"/params_"+'{0:05}'.format(i)
        print("reading "+inputDataFile)
        temp_df=pd.read_csv(inputDataFile+"_RESULT.csv")
        if(train_df.empty):
            train_df=temp_df
        else:
            train_df=train_df.append(temp_df, ignore_index=True)
        lastFileIndex=i

    train_df = train_df.drop_duplicates()
    X_train = train_df.drop(['CSI','biomass1_SS','biomass2_SS', 'biomass1', 'biomass2'], axis=1)
    if 'biomass3' in train_df.columns:
        X_train = train_df.drop(['biomass3_SS'], axis=1)
    y_train = train_df['CSI'] 
    return X_train, y_train, lastFileIndex

def generateRangesScalar(paramsRangeFile):
    paramsRangeFileDf=pd.read_csv(paramsRangeFile)
    minValueRange=paramsRangeFileDf['MinValue'].tolist()
    maxValueRange=paramsRangeFileDf['MaxValue'].tolist()
    scaler=[MinMaxScaler() for i in range(len(minValueRange))]
    [scaler[i].fit([[minValueRange[i]], [maxValueRange[i]]]) for i in range(len(minValueRange))]
    return minValueRange, maxValueRange, scaler, paramsRangeFileDf

def generateSurrogate(X_train, y_train, scaler):
    surrogate = DyMMMSurrogateModel(X_train.shape[1])
    print(X_train.columns.tolist())
    X_train_n=np.copy(X_train)
    for i in range(X_train.shape[1]):
        v=X_train.iloc[:,i].to_numpy()
        v=scaler[i].transform(v.reshape(-1,1))
        X_train_n[:,i]=v.reshape(v.shape[0],)           
        #X_train_n[:,i]=scaler[i].transform([X_train.iloc[:,i].to_numpy()])
    print(X_train_n)
    print(X_train_n.shape)
    print(y_train.shape)
    surrogate.train(X_train_n,y_train.to_numpy())
    return surrogate

if __name__ == '__main__':

    analysisDir=settings.simSettings["analysisDir"]
    communityName=settings.simSettings["communityName"]
    paramsRangeFile=analysisDir+"/screening_inputparams.csv"


    if(len(sys.argv)>1):
        analysisDir=sys.argv[1]
        paramsRangeFile=analysisDir+"/screening_inputparams.csv"
        
    if(len(sys.argv)>2):
        communityName=sys.argv[2]

    #create range and data scaler
    minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(paramsRangeFile)

    #generate data for surrogate training
    X_train, y_train, lastFileIndex = generateTrainingDF(analysisDir)

    #generate surrogate model
    surrogate=generateSurrogate(X_train, y_train, scaler)
    
    numberOfParams=paramsRangeDf.shape[0]
    #generate adaptive samples
    sampler=DyMMMSamplesAdaptive(numberOfParams, minValueRange, maxValueRange, scaler, surrogate)
    res, ps, pf = sampler.sample(X_train, y_train, X_train.shape[0])

    #save generated adaptive samples as params for nex round of simulations
    df = pd.DataFrame(data=res.X, columns=X_train.columns)
    paramDataFile = analysisDir+"/params_"+'{0:05}'.format(lastFileIndex+1)+".csv"
    if not os.path.exists(paramDataFile):
        df.to_csv(paramDataFile, index=False)
    else:
        print("ERROR - {} already exists".format(paramDataFile))



