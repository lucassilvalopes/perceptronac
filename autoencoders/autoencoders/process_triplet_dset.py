import os
import sys
import numpy as np


def get_folder_leaves_os_walk(start_path):
    folder_leaves = []
    for root, dirs, _ in os.walk(start_path):
        if not dirs:  # If 'dirs' is empty, it's a folder leaf
            folder_leaves.append(root)
    return folder_leaves


if __name__ == "__main__":

    src = "/home/lucaslopes/vimeo_triplet"

    lst = get_folder_leaves_os_walk(src)
    for pth in lst:

        fl = os.path.join(pth,"im1.png")

        new_fl = fl.replace("vimeo_triplet_old","vimeo_singlet")

        break_pt = new_fl.index("vimeo_singlet/")+len("vimeo_singlet/")
        new_fl = new_fl[:break_pt] + new_fl[break_pt:].replace("/","_") 
        
        os.rename(fl, new_fl)