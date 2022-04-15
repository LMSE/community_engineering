
import sys
import pandas as pd
import DyMMMReport

if __name__ == '__main__':

    communityName="communitypred"

    if(len(sys.argv)>1):
        communityName=sys.argv[1]

    outFile="data/"+communityName

    #Community plot
    DyMMMReport.report1(outFile+"_HUSER")
    #flux plots
    DyMMMReport.report1(outFile+"_TUSER")

