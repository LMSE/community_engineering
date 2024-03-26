import os
import numpy as np
import DyMMMSettings as settings



communitiesDir=settings.simSettings["communitiesDir"]
communityName="communitycomp"

X=None
for i in range(8552):
    fileName="{}/comp/{}_eigen_{}.txt".format(communitiesDir,communityName,str(i))
    if os.path.exists(fileName):
        test=np.loadtxt(fileName).view(complex)
    else:
        print("missing "+fileName)
    if i == 0:
        X=test
        continue
    X=np.vstack((X, test))
print(X.shape)
np.savetxt("{}/{}_eigen.txt".format(communitiesDir, communityName), X.view(float))
