import numpy as np
import sys

if __name__ == "__main__":

    X_path = sys.argv[1]

    y_path = sys.argv[2]

    Xy_path = sys.argv[3]

    X = np.genfromtxt(X_path)

    y = np.genfromtxt(y_path)

    y = y.reshape(-1,1)

    Xy = np.concatenate([X,y],axis=1)

    np.savetxt(Xy_path, Xy, fmt="%d")