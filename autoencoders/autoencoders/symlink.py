import os
import sys
import numpy as np


if __name__ == "__main__":

    src = sys.argv[1]
    directory = sys.argv[2]
    dset_nickname = sys.argv[3]

    if not os.path.exists(directory):
        os.makedirs(directory)

    for f in os.listdir(src):
        os.symlink(os.path.join(src,f),f"{directory}/symlink_{dset_nickname}_{f}")