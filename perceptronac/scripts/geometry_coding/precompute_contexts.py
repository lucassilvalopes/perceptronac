
from perceptronac.utils import causal_context_many_pcs
import numpy as np
import sys
import os

if __name__ == "__main__":

    N = int(sys.argv[1]) # N (current level)
    M = int(sys.argv[2]) # M (previous level)
        
    for i in range(len(sys.argv)-3):
        pth = sys.argv[i+3] # path to the point cloud

        basename_without_extension = os.path.splitext(os.path.basename(pth))[0] 

        dirname = os.path.dirname(pth)

        y,X = causal_context_many_pcs([pth], N+M, M/(N+M))

        Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)

        np.savez_compressed(os.path.join(dirname,f"{basename_without_extension}_N{N}_M{M}_contexts"),Xy)