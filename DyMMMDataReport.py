
import pandas as pd

from pandas_profiling import ProfileReport


def report1(inputFile, indexCol=None):
    

    df=pd.read_csv(inputFile+".csv", index_col=indexCol)

    profile = ProfileReport(df, title=inputFile, explorative=True)
    profile.to_file(inputFile)    
