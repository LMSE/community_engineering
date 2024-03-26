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

class DyMMMSurrogateModelKeras:
    """
    A class for building and training surrogate models using Keras for predictive modeling.

    Attributes:
        modelList (list): List of trained surrogate models.
        resModelList (list): List of models trained on residuals for enhanced prediction.
        X (numpy.ndarray): Input features for training the models.
        y (numpy.ndarray): Target outputs for training the models.
        analysisDir (str): Directory for saving analysis results.

    Methods:
        set_training_values(X, y): Sets the training data for the surrogate models.
        train(): Trains the surrogate models using the set training data.
        generateSurrogateModels(X, y): Generates surrogate models using K-Fold cross-validation.
        createModel(X_train, y_train, X_test, y_test): Creates and trains a neural network model.
        getResiduals(surrogateModels, X_test, y_test): Computes residuals between the predictions and actual values.
        predict_values(X): Predicts output values for a given input using the trained models.
        predict_variances(X): Predicts the variance of the output for a given input using the models trained on residuals.
        predictFromModel(models, X): Predicts output using an ensemble of models for a given input.
    """

    modelList=[]
    resModelList=[]
    X=None
    y=None
    analysisDir=settings.simSettings["analysisDir"]

    def set_training_values(self, X, y):
        self.X=X
        self.y=y

    def train(self):
        self.modelList=self.generateSurrogateModels(self.X, self.y)
        for model in self.modelList:
            print(model)
        self.y_res=self.getResiduals(self.modelList, self.X, self.y)
        self.resModelList=self.generateSurrogateModels(self.X, self.y_res)

    def generateSurrogateModels(self, X, y):
        """
        Generates surrogate models using K-Fold cross-validation.

        Parameters:
            X (numpy.ndarray): Input features for training.
            y (numpy.ndarray): Target outputs for training.

        Returns:
            list: A list of trained surrogate models.
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
        Creates and trains a neural network model.

        Parameters:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target outputs.
            X_test (numpy.ndarray): Testing input features.
            y_test (numpy.ndarray): Testing target outputs.

        Returns:
            Model: A trained neural network model.
        """
        #model=self.createModel_1(X_train, y_train, X_test, y_test)
        model=self.createModel_2(X_train, y_train, X_test, y_test)
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


    def getResiduals(self, surrogateModels, X_test, y_test)
        """
        Computes residuals between the predictions and actual values.

        Parameters:
            surrogateModels (list): The surrogate models used for prediction.
            X_test (numpy.ndarray): The input data for prediction.
            y_test (numpy.ndarray): The actual output values for comparison.

        Returns:
            numpy.ndarray: An array of residuals.
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
        Predicts output values for a given input using the trained models.

        Parameters:
            X (numpy.ndarray): Input features for prediction.

        Returns:
            numpy.ndarray: Predicted output values.
        """
        return self.predictFromModel(self.modelList, X)

    def predict_variances(self, X):
        """
        Predicts the variance of the output for a given input using the models trained on residuals.

        Parameters:
            X (numpy.ndarray): Input features for variance prediction.

        Returns:
            numpy.ndarray: Predicted output variances.
        """
        return self.predictFromModel(self.resModelList, X)

    def predictFromModel(self, models, X):
        """
        Predicts output using an ensemble of models for a given input.

        Parameters:
            models (list): The ensemble of models to use for prediction.
            X (numpy.ndarray): The input features for prediction.

        Returns:
            numpy.ndarray: The average predicted output from the ensemble of models.
        """
        y_pred=np.zeros(shape=(X.shape[0],len(models)))
        i = 0
        for model in models:
            y_pred[:,i]=model.predict(X).reshape(X.shape[0])
            i+=1
        return(y_pred.mean(axis=1))
