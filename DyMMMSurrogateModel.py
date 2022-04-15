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

    surrogateModel=None
    r2List=[]
    rmseList=[]
    analysisDir=settings.simSettings["analysisDir"]


    def __init__(self, paramCount=18):
        #self.surrogateModel = KPLS(theta0=[1e-2], poly='quadratic', corr='abs_exp', n_comp=paramCount)
        #self.surrogateModel=DyMMMSurrogateModelKeras()
        self.surrogateModel=DyMMMSurrogateModelCluster()
        #self.randomClassifier=RandomForestClassifier(max_depth=5, n_estimators=100, max_features=17)

    def train(self, X, y):
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
        y = self.surrogateModel.predict_values(X)
        return y

    def predict_variances(self, X):
        y = self.surrogateModel.predict_variances(X)
        return y

    def test(self, X, y_true):
        
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
