
import sys
import pandas as pd
import DyMMMDataPlot
import DyMMMSettings as settings


"""
generates different plots using functions defined in DyMMMDataPlot
uncomment the code for the required plot
"""


def showSimulationPlot():
    communityName="communitycomp"

    if(len(sys.argv)>1):
        communityName=sys.argv[1]

    outFile="data/"+communityName+"_sampled"

    #Community plot
    DyMMMDataPlot.plot1(None,communityName,outFile+".csv")

    #flux plots
    DyMMMDataPlot.plot2(communityName+"-HUSER",outFile+"_HUSER.csv")
    DyMMMDataPlot.plot2(communityName+"-TUSER",outFile+"_TUSER.csv")


def showViolinPlot(column, Ylabel):
    DyMMMDataPlot.plotViolin(column, Ylabel)

if __name__ == '__main__':
    
    #show CSI
    DyMMMDataPlot.plotViolin()

    #plot biomass
    #DyMMMDataPlot.plotViolin('biomass1', 'biomass', "species 1")
    #DyMMMDataPlot.plotViolin('biomass2', 'biomass', "species 2")

    #plot all states
    str= " Diversity Factor < 0.2 "
    # DyMMMDataPlot.plotViolin('A1', ' ', 'A1 '+str)
    # DyMMMDataPlot.plotViolin('A2', ' ','A2 '+str)
    # DyMMMDataPlot.plotViolin('R1', ' ','R1 '+str)
    # DyMMMDataPlot.plotViolin('G1', ' ','G1 '+str)
    # DyMMMDataPlot.plotViolin('R2', ' ','R2 '+str)
    # DyMMMDataPlot.plotViolin('G2', ' ','G2 '+str)
    # DyMMMDataPlot.plotViolin('EX_trp__L_e', ' ', 'EX_trp__L_e  '+str)
    # DyMMMDataPlot.plotViolin('EX_his__L_e', ' ', 'EX_his__L_e  '+str)



#-----------------------------------------------------------------------------------------------

    # analysisDir=settings.simSettings["analysisDir"]
    # communityName=settings.simSettings["communityName"]
    # paramsRangeFile=analysisDir+"/screening_inputparams.csv"
    # communitiesDir=settings.simSettings["communitiesDir"]
    # communityName="communitypred_cstr"
    # inputFile = communitiesDir+"/"+communityName
    # paramFile = inputFile+"_param.csv"
    # resultFile = inputFile+"_RESULT.csv"
    # param_df=pd.read_csv(paramFile)
    # result_df=pd.read_csv(resultFile)
    # param_df['CSI']=result_df['CSI']

    # DyMMMDataPlot.showScatterMatrix(param_df)

#-----------------------------------------------------------------------------------------------
    # analysisDir=settings.simSettings["analysisDir"]
    # communityName=settings.simSettings["communityName"]
    # paramsRangeFile=analysisDir+"/screening_inputparams.csv"
    # communitiesDir=settings.simSettings["communitiesDir"]
    # communityName="communitycomp_cstr"
    # inputFile = communitiesDir+"/"+communityName
    # paramFile = inputFile+"_param.csv"
    # resultFile = inputFile+"_RESULT.csv"
    # param_df=pd.read_csv(paramFile)
    # result_df=pd.read_csv(resultFile)
    # param_df['CSI']=result_df['CSI']

    # rangeDF=None
    # rangeDF=pd.read_csv(communitiesDir+"/"+communityName+"_range.csv")
    # DyMMMDataPlot.showScatterMatrix(param_df, rangeDF)

#----------------------------------------------------------------------------------------------
    #DyMMMDataPlot.plotHyperVolume()
#----------------------------------------------------------------------------------------------

    communitiesDir=settings.simSettings["communitiesDir"]


    #DyMMMDataPlot.plotComplexNumbers(communitiesDir, "communitycoop", "communitycoop biomass 2 eigen  ", 2, False)

    #DyMMMDataPlot.plotComplexNumbers(communitiesDir, "communitycomp", "communitycomp biomass 2 eigen  ", 2, False)

    #DyMMMDataPlot.plotComplexNumbers(communitiesDir, "communitypred", "communitypred biomass 2 eigen  ", 2, False)


    
    #DyMMMDataPlot.plotComplexNumbers(communitiesDir, "communitycoop", "communitycoop biomass 2 eigen Diversity Factor > 0.9 ", 2, True)

    #DyMMMDataPlot.plotComplexNumbers(communitiesDir, "communitycomp", "communitycomp biomass 2 eigen Diversity Factor > 0.9 ", 2, True)

    #DyMMMDataPlot.plotComplexNumbers(communitiesDir, "communitypred", "communitypred biomass 2 eigen Diversity Factor > 0.9 ", 2, True)


    