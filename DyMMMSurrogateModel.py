import numpy as np
from smt.surrogate_models import KPLS

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from DyMMMSurrogateModelCluster import DyMMMSurrogateModelCluster
from DyMMMSurrogateModelKeras import DyMMMSurrogateModelKeras
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import DyMMMSettings as settings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score



class DyMMMSurrogateModel:
    """
    A class that encapsulates a surrogate model for multi-objective optimization.

    Attributes:
        surrogateModel (DyMMMSurrogateModelCluster): The surrogate model used for predictions.
        r2List (list): A list storing R-squared values for the model's predictions.
        rmseList (list): A list storing RMSE values for the model's predictions.
        analysisDir (str): Directory for saving analysis results.
    
    Methods:
        train(X, y): Trains the surrogate model using the provided data.
        predict(X): Predicts the output using the surrogate model for the given input.
        predict_variances(X): Predicts the variances for the given input.
        test(X, y_true): Tests the model with the given input and true output, logging the results.
    """

    surrogateModel=None
    r2List=[]
    rmseList=[]
    analysisDir=settings.simSettings["analysisDir"]


    def __init__(self, paramCount=18):
        """
        Initializes the DyMMMSurrogateModel with a specific surrogate model.

        Parameters:
            paramCount (int): Number of parameters for the surrogate model (default is 18).
        """
        #self.surrogateModel = KPLS(theta0=[1e-2], poly='quadratic', corr='abs_exp', n_comp=paramCount)
        #self.surrogateModel=DyMMMSurrogateModelKeras()
        self.surrogateModel=DyMMMSurrogateModelCluster()
        #self.randomClassifier=RandomForestClassifier(max_depth=5, n_estimators=100, max_features=17)

    def train(self, X, y):
        """
        Trains the surrogate model with the provided input and output data.

        Parameters:
            X (numpy.ndarray): Input data for training.
            y (numpy.ndarray): Output data for training.
        """
        print("TRAIN - START======================================")
        print(X)
        print("------------------------------------------------------")
        print(y)
        print("TRAIN - END======================================") 

        """
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        y_train=encoded_Y
        self.rfscore = cross_val_score(self.randomClassifier, X, y_train, cv=50)
        """
        self.surrogateModel.set_training_values(X, y)
        self.surrogateModel.train()

    def predict(self, X):
        """
        Predicts the output using the surrogate model for the given input.

        Parameters:
            X (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted output.
        """
        y = self.surrogateModel.predict_values(X)
        return y

    def predict_variances(self, X):
        """
        Predicts the variances for the given input using the surrogate model.

        Parameters:
            X (numpy.ndarray): Input data for variance prediction.

        Returns:
            numpy.ndarray: Predicted variances.
        """
        y = self.surrogateModel.predict_variances(X)
        return y

    def test(self, X, y_true):
        """
        Tests the model with the provided input and true output, evaluates performance, and logs the results.

        Parameters:
            X (numpy.ndarray): Input data for testing.
            y_true (numpy.ndarray): True output data for comparison.

        Returns:
            list: A list containing R-squared, RMSE, and maximum absolute error values.
        """        
        y_pred = self.predict(X)
        y_pred_var = self.predict_variances(X)

        print("PREDICTED - START==========XXX============================")
        print(X)
        print("--------------------YTRUE----------------------------------")
        print(y_true)
        print("--------------------YPRED----------------------------------")
        print(y_pred)
        print("---------------------YVAR---------------------------------")
        print(y_pred_var)
        print("PREDICTED - END======================================")
        r2 = r2_score(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")  
        self.r2List.append(r2)      
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False, multioutput="uniform_average")
        self.rmseList.append(rmse)
        abse = np.max(np.fabs(y_pred-y_true))
        print([r2, rmse, abse])
        logFile = self.analysisDir+"/RMSE2.log"
        f = open(logFile, "a")
        f.write("=======YTRUE==========\n")
        f.write(str(y_true)) 
        f.write("=======YPRED==========\n") 
        f.write(str(y_pred))
        f.write("=======YVAR==========\n") 
        f.write(str(y_pred_var))
        f.close()
        logFile = self.analysisDir+"/RMSE.log"
        f = open(logFile, "a")
        f.write("=======MODEL RMSE/ABSE=S=========\n")
        f.write(str(rmse))
        f.write(str(abse))   
        f.write("=======MODEL RMSE/ABSE=E=========\n")
        f.close()

        """
        encoder = LabelEncoder()
        encoder.fit(y_true)
        encoded_Y = encoder.transform(y_true)
        y_test=encoded_Y

        y_pred_rf=self.randomClassifier.predit(X)
        y_pred_prb=self.randomClassifier.predict_proba(X)
        logFile = self.analysisDir+"/classifier.log"
        f = open(logFile, "a")
        accuracy=accuracy_score(y_test, y_pred_rf, normalize=True)
        print("Accuracy (train) for %s: %0.1f%% " % (str(self.rfscore), accuracy * 100))
        print(y_pred_prb)
        f.write("Accuracy (train) for %s: %0.1f%% " % (str(self.rfscore), accuracy * 100))
        f.write(str(y_pred_prb))
        f.close()
        """

        return [r2, rmse, abse]
