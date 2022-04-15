import os
import sys
import numpy as np
import glob
import pandas as pd
import DyMMMSettings as settings
from DyMMMMultiObjectiveProblem import DyMMMMultiObjectiveProblem
from sklearn.preprocessing import MinMaxScaler
from DyMMMSurrogateModel import DyMMMSurrogateModel
import smtplib


def generateTestDF(inputDataDir):
    files=glob.glob(inputDataDir+"/*_RESULT.csv")
    train_df = pd.DataFrame()
    n=len(files)
    lastFileIndex=0
    for i in range(n-1):
        inputDataFile = inputDataDir+"/params_"+'{0:05}'.format(i)
        print("reading "+inputDataFile)
        temp_df=pd.read_csv(inputDataFile+"_RESULT.csv")
        if(train_df.empty):
            train_df=temp_df
        else:
            train_df=train_df.append(temp_df, ignore_index=True)
        lastFileIndex=i

    test_df_file = inputDataDir+"/params_"+'{0:05}'.format(lastFileIndex+1)
    test_df=pd.read_csv(test_df_file+"_RESULT.csv")
    lastFileIndex+=lastFileIndex

    train_df = train_df.drop_duplicates()
    X_train = train_df.drop(['CSI','biomass1_SS','biomass2_SS', 'biomass1', 'biomass2'], axis=1)
    if 'biomass3' in train_df.columns:
        X_train = train_df.drop(['biomass3_SS'], axis=1)
    y_train = train_df['CSI'] 

    test_df = test_df.drop_duplicates()
    X_test = test_df.drop(['CSI','biomass1_SS','biomass2_SS', 'biomass1', 'biomass2'], axis=1)
    if 'biomass3' in test_df.columns:
        X_test = test_df.drop(['biomass3_SS'], axis=1)
    y_test = test_df['CSI'] 

    return X_train, y_train, X_test, y_test, lastFileIndex

def generateRangesScalar(paramsRangeFile):
    paramsRangeFileDf=pd.read_csv(paramsRangeFile)
    minValueRange=paramsRangeFileDf['MinValue'].tolist()
    maxValueRange=paramsRangeFileDf['MaxValue'].tolist()
    scaler=[MinMaxScaler() for i in range(len(minValueRange))]
    [scaler[i].fit([[minValueRange[i]], [maxValueRange[i]]]) for i in range(len(minValueRange))]
    return minValueRange, maxValueRange, scaler, paramsRangeFileDf

def generateSurrogate(X_train, y_train, scaler):
    surrogate = DyMMMSurrogateModel(X_train.shape[1])
    X_train_n=np.copy(X_train)
    print(X_train.shape[1])
    for i in range(X_train.shape[1]):
        print(i)
        v=X_train.iloc[:,i].to_numpy()
        v=scaler[i].transform(v.reshape(-1,1))
        X_train_n[:,i]=v.reshape(v.shape[0],)
    print(X_train_n)
    surrogate.train(X_train_n,y_train.to_numpy())
    return surrogate

if __name__ == '__main__':
    analysisDir=settings.simSettings["analysisDir"]
    communityName=settings.simSettings["communityName"]
    paramsRangeFile=analysisDir+"/screening_inputparams.csv"

    #create range and data scaler
    minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(paramsRangeFile)

    #generate data for surrogate training
    X_train, y_train, X_test, y_test, lastFileIndex = generateTestDF(analysisDir)

    #generate surrogate model
    X_train.to_csv(analysisDir+"/X_train.csv", index=False)
    y_train.to_csv(analysisDir+"/y_train.csv", index=False)    
    surrogate=generateSurrogate(X_train, y_train, scaler)

    X_test.to_csv(analysisDir+"/X_test.csv", index=False)
    y_test.to_csv(analysisDir+"/y_test.csv", index=False)   

    X_test_n=np.copy(X_test)
    for i in range(X_test.shape[1]):
        v=X_test.iloc[:,i].to_numpy()
        v=scaler[i].transform(v.reshape(-1,1))
        X_test_n[:,i]=v.reshape(v.shape[0],)        
        #X_test_n[:,i]=scaler[i].transform([X_test.iloc[:,i].to_numpy()])

    r2, rmse, abse = surrogate.test(X_test_n, y_test.to_numpy())

    errorFile=analysisDir+"/testerror.csv"
    error_df=pd.read_csv(errorFile)
    error_df=error_df.append({'r2':r2, 'rmse':rmse, 'abse':abse}, ignore_index=True)
    error_df.to_csv(errorFile, index=False)

    # rows = error_df.shape[0]
    # values=error_df.iloc[-1,:]

    # gmail_user = '@gmail.com'
    # gmail_password = ''

    # sent_from = gmail_user
    # to = ['@gmail.com']
    # subject = str(rows)
    # body = "{}".format(str(values))

    # email_text = """\
    # From: %s
    # To: %s
    # Subject: %s

    # %s
    # """ % (sent_from, ", ".join(to), subject, body)

    # try:
    #     server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    #     server.ehlo()
    #     server.login(gmail_user, gmail_password)
    #     server.sendmail(sent_from, to, email_text)
    #     server.close()

    #     print('Email sent!')
    # except Exception as e:
    #     print(e)
    #     print('Something went wrong...')