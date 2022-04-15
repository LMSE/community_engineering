@REM set DyMMM_communityName=communitycoop
@REM python main_runDyMMM.py
@REM set DyMMM_communityName=communitycomp
@REM python main_runDyMMM.py
@REM set DyMMM_communityName=communitypred
@REM python main_runDyMMM.py

set DyMMM_communityName=communitycoop
python main_runDyMMM_Stability.py data\communitycoop.csv 1
python main_runDyMMM_Stability.py data\communitycoop.csv 2
python main_runDyMMM_Stability.py data\communitycoop.csv 4
python main_runDyMMM_Stability.py data\communitycoop.csv 6
python main_runDyMMM_Stability.py data\communitycoop.csv 8
python main_runDyMMM_Stability.py data\communitycoop.csv 10

set DyMMM_communityName=communitycomp
python main_runDyMMM_Stability.py data\communitycomp.csv 1
python main_runDyMMM_Stability.py data\communitycomp.csv 2
python main_runDyMMM_Stability.py data\communitycomp.csv 4
python main_runDyMMM_Stability.py data\communitycomp.csv 6
python main_runDyMMM_Stability.py data\communitycomp.csv 8
python main_runDyMMM_Stability.py data\communitycomp.csv 10


set DyMMM_communityName=communitypred
python main_runDyMMM_Stability.py data\communitypred.csv 1
python main_runDyMMM_Stability.py data\communitypred.csv 2
python main_runDyMMM_Stability.py data\communitypred.csv 4
python main_runDyMMM_Stability.py data\communitypred.csv 6
python main_runDyMMM_Stability.py data\communitypred.csv 8
python main_runDyMMM_Stability.py data\communitypred.csv 10


set DyMMM_communityName=communitycomp
python main_runDyMMM_Stability.py data\communitycomp.csv 1
python main_runDyMMM_Stability.py data\communitycomp.csv 2

set DyMMM_communityName=communitypred
python main_runDyMMM_Stability.py data\communitypred.csv 1
python main_runDyMMM_Stability.py data\communitypred.csv 2

set DyMMM_communityName=communitycoop
python main_runDyMMM_Stability.py data\communitycoop.csv 1
python main_runDyMMM_Stability.py data\communitycoop.csv 2
