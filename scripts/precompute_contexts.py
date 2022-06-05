
from perceptronac.utils import causal_context_many_pcs
import numpy as np
import sys
import os

if __name__ == "__main__":

    N = int(sys.argv[1]) # N (context size)
        
    for i in range(len(sys.argv)-2):
        pth = sys.argv[i+2] # path to the point cloud

        basename_without_extension = os.path.splitext(os.path.basename(pth))[0] 

        dirname = os.path.dirname(pth)

        y,X = causal_context_many_pcs([pth], N, 0)

        Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)

        np.savez_compressed(os.path.join(dirname,f"{basename_without_extension}_N{N}_contexts"),Xy)