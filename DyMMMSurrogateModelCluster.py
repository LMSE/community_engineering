import numpy as np
import math
import pickle
import types
import tempfile
import pandas as pd
import statistics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import tensorflow_probability as tfp
#import tensorflow_datasets as tfds
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import sklearn
from sklearn import svm
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import kneighbors_graph


import DyMMMSettings as settings
from tensorflow import random

np.random.seed(2017)
random.set_seed(1)

def root_mean_squared_error(y_true, y_pred): 
        RMSE=K.sqrt(K.mean(K.square(y_pred - y_true)))
        return  RMSE

""" 
#linux OS
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            print("---------------------1111111111111111111----------------------------------------------------"+fd.name)
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__
    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__ 
"""
"""
#windows OS
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(delete=True) as fd:
            print("---------------------1111111111111111111----------------------------------------------------"+fd.name)
            #fd.close()
            keras.models.save_model(self, fd.name)
            with open(fd.name,'rb') as fd:
                model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(delete=True) as fd:
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)
            print("--------------------22222222222222222222222222-----------------------------------------------------"+fd.name)

            #fd.close()
            with open(fd.name,'wb') as fd:
                fd.write(state['model_str'])
                fd.flush()
                model = keras.models.load_model(fd.name)
            self.__dict__ = model.__dict__
    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
"""
 
#https://stackoverflow.com/questions/45576576/keras-unknown-loss-function-error-after-defining-custom-loss-function


def unpack(model, training_config, weights):
    get_custom_objects().update({'root_mean_squared_error': root_mean_squared_error})
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():


    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))


    cls = Model
    cls.__reduce__ = __reduce__


make_keras_picklable()

class DyMMMSurrogateModelCluster:
    """
    A class for clustering-based surrogate modeling, using various machine learning techniques
    to create and manage a set of surrogate models based on cluster assignments.
    
    Attributes:
        modelList (list): List of all models.
        resModelList (list): List of residual models.
        X (numpy.ndarray): Training input data.
        y (numpy.ndarray): Training output data.
        analysisDir (str): Directory for storing analysis data.
        modelsInSurrogate (list): List of models for each cluster.
        modelsInSurrogateType (list): Type of model used for each cluster.
        cluster_assigner (xgboost.XGBClassifier): Model to assign new points to a cluster.
        clusters_X (list): List of input data points for each cluster.
        clusters_y (list): List of output data points for each cluster.
        
    Methods:
        set_training_values(X, y): Sets the training data.
        train(): Trains the surrogate models based on clustering.
        generateSurrogateModels(X, y): Generates surrogate models for a cluster.
        createModel(X_train, y_train, X_test, y_test): Creates a surrogate model.
        getResiduals(surrogateModels, X_test, y_test): Calculates the residuals of predictions.
        predict_values(X): Predicts output values for given input.
        predict_variances(X): Predicts variances for given input.
        predictFromModel(models, X): Predicts a value from an ensemble of models.
    """

    modelList=[]
    resModelList=[]
    X=None
    y=None
    analysisDir=settings.simSettings["analysisDir"]

    def set_training_values(self, X, y):
        """
        Sets the training values for the model.

        Parameters:
            X (numpy.ndarray): Input features for training.
            y (numpy.ndarray): Output values for training.
        """
        self.X=X
        self.y=y

    def train(self):

        X_train=self.X
        y_train=self.y

        connectivity = kneighbors_graph(X_train, n_neighbors=25, mode='connectivity', include_self='auto', metric='cityblock', p=1)
        
        clusterCount=int(X_train.shape[0]/(X_train.shape[1]*100))
        if clusterCount < 1:
            clusterCount=1
        #model = AgglomerativeClustering(n_clusters=clusterCount, affinity='manhattan',  linkage='complete')
        model = AgglomerativeClustering(connectivity=connectivity, n_clusters=clusterCount, compute_full_tree=True)


        y_cluster_predict=model.fit_predict(X_train)
        print(model.n_clusters_)

        clusters_X=[ [] for _ in range(clusterCount) ]
        clusters_y=[ [] for _ in range(clusterCount) ]

        modelsInSurrogate=[ [] for _ in range(clusterCount) ]
        modelsInSurrogateType=[ [] for _ in range(clusterCount) ]
        resModelList=[ [] for _ in range(clusterCount) ]

        for index,row in enumerate(X_train):
            clusters_X[y_cluster_predict[index]].append(X_train[index])
            clusters_y[y_cluster_predict[index]].append(y_train[index])

        print("-------------------------------------------Number of points in each cluster-----------------------------------")
        for index in range(clusterCount):
            print(len(clusters_X[index]))


        for index in range(clusterCount):
            X = np.asarray(clusters_X[index], dtype=np.float32)
            y = np.asarray(clusters_y[index], dtype=np.float32)      
            print(len(clusters_X[index]))

            if(X.shape[0]<0):
                #modelsInSurrogate[index]=None
                modelsInSurrogate[index] = RandomForestRegressor(max_depth=2, random_state=0).fit(X,y.ravel())
                modelsInSurrogateType[index]=0
            #elif(X.shape[0]<1720000):
            else:
                modelsInSurrogate[index]=self.generateSurrogateModels(X, y)
                modelsInSurrogateType[index]=1
            """
            elif(X.shape[0]<172000):
                modelsInSurrogate[index]=self.generateSurrogateModels(X, y)
                modelsInSurrogateType[index]=2
            else:
                modelsInSurrogate[index] = KPLS(theta0=[1e-2], poly='quadratic', corr='abs_exp', n_comp=paramCount)
                modelsInSurrogate[index].set_training_values(X, y)
                modelsInSurrogate[index].train()
                modelsInSurrogateType[index]=3
            """

            y_res=self.getResiduals(modelsInSurrogate[index], X, y)
            resModelList[index]=self.generateSurrogateModels(X, y_res)


        encoder = LabelEncoder()
        encoder.fit(y_cluster_predict)
        y_cluster_predict = encoder.transform(y_cluster_predict)
        
        X1_train, X1_test, y1_train, y1_test = train_test_split(X_train, y_cluster_predict, test_size=0.33, random_state=42)

        param_dist = {'objective':'multi:softprob', 'n_estimators':100, 'num_class':clusterCount, 'max_depth':17, 'eval_metric':'mlogloss'}

        cluster_assigner = xgb.XGBClassifier(**param_dist)
        
    
        cluster_assigner.fit(X1_train, y1_train,
                eval_set=[(X1_train, y1_train), (X1_test, y1_test)],
                #eval_metric='mlogloss',
                verbose=True)

        evals_result = cluster_assigner.evals_result()

        self.modelsInSurrogate=modelsInSurrogate
        self.modelsInSurrogateType=modelsInSurrogateType
        self.cluster_assigner=cluster_assigner
        self.clusters_X=clusters_X
        self.clusters_y=clusters_y
        self.modelsInSurrogate=modelsInSurrogate
        self.modelsInSurrogateType=modelsInSurrogateType
        self.resModelList=resModelList


    def generateSurrogateModels(self, X, y):
        """
        Generates surrogate models for a specific cluster.

        Parameters:
            X (numpy.ndarray): Cluster input features.
            y (numpy.ndarray): Cluster output values.
        
        Returns:
            list: A list of trained surrogate models for the cluster.
        """
        kf = KFold(n_splits=max(int(X.shape[0]/200),2))
        surrogateModels=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            surrogateModels.append(self.createModel(X_train, y_train, X_test, y_test))
        print("-----------generateSurrogateModels------------")
        for model in surrogateModels:
            print(model)
        return surrogateModels


    def createModel(self, X_train, y_train, X_test, y_test):
        """
        Creates and trains a surrogate model.

        Parameters:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training output values.
            X_test (numpy.ndarray): Testing input features.
            y_test (numpy.ndarray): Testing output values.
        
        Returns:
            model: A trained surrogate model.
        """
        #model=self.createModel_1(X_train, y_train, X_test, y_test)
        #model=self.createModel_2(X_train, y_train, X_test, y_test)
        model= svm.NuSVR(gamma='auto')
        model.fit(X_train, y_train.ravel())
        return(model)

    def createModel_1(self, X_train, y_train, X_test, y_test):

        learning_rate = 1e-5
        num_epochs = 10000


        mse_loss = keras.losses.MeanSquaredError()

        # inputs = keras.Input(shape=X_train.shape[1], name="digits")
        # x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        # x = layers.Dense(64, activation="relu", name="dense_2")(x)
        # outputs = layers.Dense(1, name="predictions")(x)

        dropout_rate = 0.1
        inputs = Input(shape=X_train.shape[1])
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(dropout_rate)(x, training=True)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x, training=True)
        outputs = Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss=mse_loss,
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

        earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                            mode ="min", patience = 5,  
                                            restore_best_weights = True) 

        print("Start training the model...")
        model.fit(X_train, y_train , batch_size=32, epochs=num_epochs, validation_data=(X_test, y_test), callbacks =[earlystopping])
        print("Model training finished.")
        _, rmse = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test RMSE: {round(rmse, 3)}")
        logFile = self.analysisDir+"/RMSE.log"
        f = open(logFile, "a")
        f.write("=======MODEL RMSE==========\n")
        f.write(str(rmse))
        f.close()
        return model

    def createModel_2(self, X_train, y_train, X_test, y_test):

        learning_rate = 1e-5
        num_epochs = 10000

        #initializer = tf.keras.initializers.Orthogonal()
        initializer= tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        #bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)
        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=None)

        # define model
        model = Sequential()
        model.add(Dense(16, input_dim=17, activation='relu', kernel_initializer=initializer, use_bias=True, bias_initializer=bias_initializer))
        #model.add(Dense(64,  activation='relu', kernel_initializer=initializer))
        model.add(Dense(32,  activation='relu', kernel_initializer=initializer))
        model.add(Dense(32,  activation='relu', kernel_initializer=initializer))
        model.add(Dense(64,  activation='relu', kernel_initializer=initializer))
        model.add(Dense(32,  activation='relu', kernel_initializer=initializer))
        model.add(Dense(32,  activation='relu', kernel_initializer=initializer))
        #model.add(Dense(16,  activation='relu', kernel_initializer=initializer ))
        model.add(Dense(8,  activation='relu', kernel_initializer=initializer))
        model.add(Dense(1))

        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=0.0001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
            name='RMSprop'
        )

        model.compile(loss=root_mean_squared_error, optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])

        earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                            mode ="min", patience = 50,  
                                            restore_best_weights = True) 

        print("Start training the model...")
        model.fit(X_train, y_train , batch_size=32, epochs=num_epochs, validation_data=(X_test, y_test), callbacks =[earlystopping])
        print("Model training finished.")
        _, rmse = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test RMSE: {round(rmse, 3)}")
        logFile = self.analysisDir+"/RMSE.log"
        f = open(logFile, "a")
        f.write("=======MODEL RMSE==========\n")
        f.write(str(rmse))
        f.close()
        print(model)
        return model


    def getResiduals(self, surrogateModels, X_test, y_test):
        """
        Computes the residuals of predictions from the surrogate models.

        Parameters:
            surrogateModels (list): List of surrogate models.
            X_test (numpy.ndarray): Testing input features.
            y_test (numpy.ndarray): Testing output values.
        
        Returns:
            numpy.ndarray: Residuals of the predictions.
        """
        print("Generating residuals...")

        y_pred=np.zeros(shape=(y_test.shape[0],len(surrogateModels)))
        y_res=np.zeros(shape=(y_test.shape[0],len(surrogateModels)))

        i = 0
        for model in surrogateModels:
            print(model)
            y_pred[:,i]=model.predict(X_test).reshape(X_test.shape[0])
            print(y_pred[:,i].shape)
            print(y_test.shape)
            y_res[:,i]=(y_pred[:,i]-y_test.reshape(y_test.shape[0]))**2
            i+=1
        print(y_res)
        print(y_res.mean(axis=1))
        return y_res.mean(axis=1)

    def predict_values(self, X):
        """
        Predicts output values for given input using the trained surrogate models.

        Parameters:
            X (numpy.ndarray): Input features for prediction.
        
        Returns:
            numpy.ndarray: Predicted output values.
        """
        y_pred = np.zeros(X.shape[0])
        predicted_cluster=self.cluster_assigner.predict(X)
        for index, row in enumerate(X):
            clusterIndex=predicted_cluster[index]-1
            y_pred[index]=self.predictFromModel(self.modelsInSurrogate[clusterIndex],X[index])
        return y_pred

    def predict_variances(self, X):
        """
        Predicts variances for given input using the residual models.

        Parameters:
            X (numpy.ndarray): Input features for prediction.
        
        Returns:
            numpy.ndarray: Predicted variances.
        """
        y_pred_var = np.zeros(X.shape[0])
        predicted_cluster=self.cluster_assigner.predict(X)
        print('predicted cluster '+str(predicted_cluster))
        for index, row in enumerate(X):
            clusterIndex=predicted_cluster[index]
            if type(clusterIndex) == np.ndarray:
                index=next((i for i, x in enumerate(clusterIndex.tolist()) if x), None)
                clusterIndex=index
            else:
                clusterIndex=0
            clusterModel=self.resModelList[clusterIndex]
            arr=self.predictFromModel(clusterModel,X[index])
            y_pred_var[index]=arr
        return y_pred_var

    """
    def predict_variances(self, X):
        y_pred_var = np.zeros(X.shape[0])
        predicted_cluster=self.cluster_assigner.predict(X)-1
        for index, row in enumerate(X):
            clusterIndexList=predicted_cluster[index]
            #clusterIndex = [x > -1 for x in clusterIndexList]
            clusterIndex=next(x[0] for x in enumerate(clusterIndexList) if x[1] > -1)
            y_pred_var[index]=self.predictFromModel(self.resModelList[clusterIndex],X[index])
        return y_pred_var
    """

    def predictFromModel(self, models, X):
        """
        Predicts an output value from an ensemble of models for a given input.

        Parameters:
            models (list): List of models to predict from.
            X (numpy.ndarray): Single input feature set.
        
        Returns:
            float: The mean of the predictions from the ensemble of models.
        """
        y_pred=np.zeros(len(models))
        i = 0
        for model in models:
            y_pred[i]=model.predict(X.reshape(1, -1))
            i+=1
        return(y_pred.mean())
