import os
import sys

if __name__ == "__main__":

    strng = sys.argv[1] # "L5e-3_"

    fldrs = []
    for i in range(2,len(sys.argv)):
        fldrs.append(sys.argv[i])

    for fldr in fldrs:
        for f in os.listdir(fldr):
            new_f = f"{strng}{f}"
            os.rename(os.path.join(fldr,f), os.path.join(fldr,new_f))