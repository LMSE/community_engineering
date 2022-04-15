import os
import pandas as pd
from DyMMMSurrogateModel import DyMMMSurrogateModel
import DyMMMSettings as settings




if __name__ == '__main__':

    analysisDir=settings.simSettings["analysisDir"]
    communityName=settings.simSettings["communityName"]
    index=0 
    CSIDF = pd.DataFrame()
    while True:
        inputFile = analysisDir+"/params_"+'{0:05}'.format(index)
        if os.path.exists(inputFile+"_RESULT.csv"):
            #print(inputFile)
            temp_df=pd.read_csv(inputFile)
            if(CSIDF.empty):
                CSIDF=temp_df
            else:
                y=CSIDF['CSI']
                df=CSIDF.copy().drop(['biomass1_SS', 'biomass2_SS','CSI'])
                surrogate = DyMMMSurrogateModel()
                surrogate.train(df.to_numpy(),y.numpy())
                y=temp_df['CSI']
                df=temp_df.copy().drop(['biomass1_SS', 'biomass2_SS','CSI'])
                r2, rmse, abse=surrogate.test(df.to_numpy(),y.numpy())
                print("{},{},{},{}".format(str(index),str(r2),str(rmse),str(abse)))
                CSIDF=CSIDF.append(temp_df, ignore_index=True)
            index+=1
        else:
            break

