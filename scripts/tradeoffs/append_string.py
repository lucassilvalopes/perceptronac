import os
import sys

if __name__ == "__main__":

    fldr = sys.argv[1]
    strng = sys.argv[2] # "L5e-3_"

    for f in os.listdir(fldr):
        new_f = f"{strng}{f}"
        os.rename(os.path.join(fldr,f), os.path.join(fldr,new_f))