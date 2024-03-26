import numpy as np
import math
from pymoo.core.problem import Problem
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler



def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


class DyMMMMultiObjectiveProblem(Problem):
    """
    Defines a multi-objective optimization problem based on data and a surrogate model. This class is designed to
    work with optimization algorithms in pymoo, providing a way to evaluate solutions based on the provided surrogate model.

    Attributes:
        localWeight (float): Weight for the local objective.
        globalWeight (float): Weight for the global objective.
        train_X (numpy.ndarray): Training data features, converted to numpy array.
        train_y (numpy.ndarray): Training data labels, converted to numpy array.
        surrogate (object): A surrogate model that supports predict_variances method.
        scaler (list): A list of scaler objects to scale input features.

    Args:
        paramCount (int): Number of parameters (design variables).
        lowerBounds (numpy.ndarray): Lower bounds for the design variables.
        upperBounds (numpy.ndarray): Upper bounds for the design variables.
        train_X (pandas.DataFrame): Training data features.
        train_y (pandas.DataFrame): Training data labels.
        scaler (list): A list of scaler objects to scale input features.
        surrogate (object): A surrogate model that supports predict_variances method.
    """
    localWeight=1.0
    globalWeight=1.0
    train_X=None
    train_y=None
    surrogate=None
    scaler=None

    def __init__(self,
                 paramCount,
                 lowerBounds,
                 upperBounds,
                 train_X,
                 train_y,
                 scaler,
                 surrogate):
        super().__init__(n_var=paramCount,
                         n_obj=2, #number of objectives
                         n_constr=0, #number of constraints
                         xl=lowerBounds, #lower bounds of the design variables
                         xu=upperBounds) #upper bounds of the design variables

        self.train_X = train_X.to_numpy()
        self.train_y = train_y.to_numpy()
        self.surrogate=surrogate
        self.scaler=scaler

    # given an input from algorithm X between bounds create a measure for out
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates the objectives for the optimization problem. The objectives are to maximize the distance from
        the training sample (f1) and to minimize the prediction error (f2) based on the surrogate model.

        Args:
            X (numpy.ndarray): The solutions to evaluate.
            out (dict): A dictionary where the evaluated objective values are stored.

        Updates:
            out["F"]: A numpy array with the evaluated objective values for each solution.
        """
        #f1= maximize distance from training sample   (https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
        #f2= minimize error
        #train_X=self.train_X.reshape(-1,1)
        
        train_X=self.train_X

        #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        #print("X.shape={}".format(str(X.shape)))
        #print(X)
        #print(train_X)
        #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        X_n=np.copy(X)
        train_X_n=np.copy(train_X)
        #X_n=np.append(X,train_X, axis=0)
        for i in range(X.shape[1]):
            v=X[:,i]
            v=self.scaler[i].transform(v.reshape(-1,1))
            X_n[:,i]=v.reshape(v.shape[0],)              
            #X_n[:,i]=self.scaler[i].transform([X[:,i]])
        for i in range(train_X.shape[1]):
            v=train_X[:,i]
            v=self.scaler[i].transform(v.reshape(-1,1))
            train_X_n[:,i]=v.reshape(v.shape[0],)                
            #train_X_n[:,i]=self.scaler[i].transform([train_X[:,i]])
        f1_array=distance.cdist(X_n, train_X_n, 'cityblock')
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(f1_array.shape)
        # print(f1_array)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # #f1=np.array(f1_array.shape[0],1)
        f1=-1.0*self.globalWeight*f1_array.min(axis=1).reshape(f1_array.shape[0],1)
        #print(X.shape)
        f2=-1.0*self.localWeight*self.surrogate.predict_variances(X_n)
        #greedy strategy
        tempvar=self.localWeight
        self.localWeight = self.globalWeight 
        self.globalWeight = tempvar
        #print("f1={} f2={} {} {} ".format(str(f1), str(f2), str(self.localWeight), str(self.globalWeight)))
        out["F"] = np.column_stack([f1, f2])
        #print(out["F"])
        # print("=======================================================================================================================================================================================")
        #print(f1.shape)
        #print(f2.shape)
        # print("=======================================================================================================================================================================================")
