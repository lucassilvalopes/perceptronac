import os
import sys

if __name__ == "__main__":

    fldr = sys.argv[1]
    lmbd = sys.argv[2]

    for f in os.listdir(fldr):
        new_f = f"L{lmbd}_{f}"
        os.rename(os.path.join(fldr,f), os.path.join(fldr,new_f))