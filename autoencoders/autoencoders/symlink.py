import os
import sys
import numpy as np


if __name__ == "__main__":

    src = sys.argv[1]

    if src.endswith('/'):
        src = src[:-1]

    dset_nickname = os.path.basename(os.path.dirname(src))

    os.mkdir("symlinks")

    for f in os.listdir(src):
        if not (f.endswith("x2.png") or f.endswith("x3.png") or f.endswith("x4.png")):
            os.symlink(os.path.join(src,f),f"symlinks/symlink_{dset_nickname}_{f}")