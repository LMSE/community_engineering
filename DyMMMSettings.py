
import psutil
import os

simSettings ={
    "communitiesDir":"../communities",
    "communityName":'communitycoop_cstr',
    "solverName":'BDF',
    "stopTime":100,
    "analysisDirName":"v8",
    "morrisDirName":"morris",
    "rangeFileName":"screening_inputparams.csv",
    "rangeFileNameKeyParams":"screening_inputparams_key.csv",
}

if 'DyMMM_communityName' in os.environ:
    simSettings["communityName"]=os.environ['DyMMM_communityName']

if 'DyMMM_analysisDirName' in os.environ:
    simSettings["analysisDirName"]=os.environ['DyMMM_analysisDirName']

if 'DyMMM_solverName' in os.environ:
    simSettings["DyMMM_solverName"]=os.environ['DyMMM_solverName']

simSettings["communityDir"]=simSettings["communitiesDir"]+"/"+simSettings["communityName"]
simSettings["analysisDir"]=simSettings["communityDir"]+"/"+simSettings["analysisDirName"]
simSettings["paramsRangeFilePath"]=simSettings["analysisDir"]+"/"+simSettings["rangeFileName"]
simSettings["keyParamsRangeFilePath"]=simSettings["analysisDir"]+"/"+simSettings["rangeFileNameKeyParams"]
simSettings["sobolParams"]=simSettings["analysisDir"]+"/sobolParams.csv"

simSettings["morrisDir"]=simSettings["communityDir"]+"/"+simSettings["morrisDirName"]
simSettings["morrisParams"]=simSettings["morrisDir"]+"/morrisParams.csv"
simSettings["morrisParamsRangeFilePath"]=simSettings["morrisDir"]+"/"+simSettings["rangeFileName"]


jobParams = {
    "partition":0,    
    "totalJobs":1,
    "paramRowsPerJob":1
}

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    #p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


if __name__ == '__main__':
    print(simSettings)