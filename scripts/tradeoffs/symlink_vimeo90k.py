import os
import sys
import numpy as np


if __name__ == "__main__":

    src = sys.argv[1]
    dst = sys.argv[2]
    max_n = None
    if len(sys.argv) > 3:
        max_n = int(sys.argv[3])
    n = 0
    p = 0.01
    for root, dirs, files in os.walk(src):
        for name in files:
            if (max_n is not None) and (np.random.random(1)[0] > p):
                continue
            item = os.path.join(root, name)
            i = item.split('/').index( os.path.basename(src) )
            os.symlink(item, os.path.join(dst,'_'.join(item.split('/')[i:])))
            n += 1
            if n == max_n:
                break
        if n == max_n:
            break